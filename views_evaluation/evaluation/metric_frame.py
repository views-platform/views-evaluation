"""
MetricFrame — the typed, transportable, provenance-stamped evaluation-of-record.

This is views-evaluation's emit-side artifact for the cross-repo chain
    views-frames (substrate) → views-evaluation (emit MetricFrame) → views-reporting (consume)
decided in views-frames ADR-020. views-evaluation OWNS the MetricFrame type and its
evaluation-specific vocabulary; views-frames provides the substrate it reuses:

    * ``FrameMetadata``                       — the generic provenance header (ADR-013)
    * ``conformance.assert_frame_envelope``   — the published structural checker (ADR-016/020)

A MetricFrame is a string-keyed value object (NOT a spatiotemporal ``(time, unit)`` frame),
keyed by the axes ``(eval_type, target, metric, group_id, partition, level)``. It exposes the
shared "frame envelope" surface (``values`` float32 with an explicit trailing axis, ``n_rows``,
and a ``save``/``load`` round-trip) so a consumer can validate it with
``views_frames.conformance.assert_frame_envelope`` rather than re-asserting drifting copies
(register C-46).

Provenance is split by concern (ADR-020 / register C-47): generic identity (model, run_id,
data_version, …) lives in the reused ``FrameMetadata``; evaluation-specific identity
(``scoring_code_version``, full-precision ``evaluation_timestamp``) stays here and must never
leak into the generic header.

Importing this module requires the optional ``views-frames`` dependency
(``pip install views-evaluation[frames]``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from views_frames import FrameMetadata

# Level-1 component: logs at ERROR before raising (logging standard §5.1).
# The Level-0 pure-math modules are exempt and must NOT acquire loggers.
logger = logging.getLogger(__name__)

# ── Vocabulary authority ────────────────────────────────────────────────────────
# The single mapping from views-evaluation's internal schema names to the
# consumer-facing eval_type spelling that views-reporting matches on.
SCHEMA_TO_EVAL_TYPE: Dict[str, str] = {
    "month": "month-wise",
    "time_series": "time-series-wise",
    "step": "step-wise",
}

# Ordered axes of the MetricFrame key space (ADR-020).
AXES = ("eval_type", "target", "metric", "group_id", "partition", "level")

# The group_id assigned to the cross-group aggregate row that views-reporting reads.
MEAN_GROUP_ID = "mean"

# Wire-schema marker. The cross-repo emit→consume contract (register C-46, open half)
# is views-evaluation's responsibility; this is its version handle.
SCHEMA_VERSION = "1.0.0"


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback: coerce numpy scalar types (e.g. an injected np.int64
    timestamp/seed) to native Python so ``save()`` never crashes on serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    err_msg = f"Object of type {type(obj).__name__} is not JSON serializable"
    logger.error(err_msg)
    raise TypeError(err_msg)


def _source_git_sha() -> Optional[str]:
    """Short SHA of the worktree this module is running from, or None.

    Read from `.git` directly rather than shelling out: `git` is not guaranteed to exist
    where evaluations run, and a missing binary must not change what gets stamped.
    """
    path = Path(__file__).resolve().parent
    for candidate in (path, *path.parents):
        git = candidate / ".git"
        if not git.exists():
            continue
        try:
            if git.is_file():  # worktree: `.git` is a file pointing at the real dir
                git = Path(git.read_text().split("gitdir:", 1)[1].strip())
            head = (git / "HEAD").read_text().strip()
            if head.startswith("ref:"):
                ref = head.split(":", 1)[1].strip()
                target = git / ref
                if target.exists():
                    return target.read_text().strip()[:7]
                packed = git / "packed-refs"
                if packed.exists():
                    for line in packed.read_text().splitlines():
                        if line.endswith(f" {ref}"):
                            return line.split()[0][:7]
                return None
            return head[:7]
        except (OSError, IndexError):
            return None
    return None


def default_scoring_code_version() -> Optional[str]:
    """The version stamped into an emitted MetricFrame's provenance.

    The installed distribution's version, plus `+g<sha>` when this module is running
    from a git worktree.

    **Why the SHA.** The version alone comes from `importlib.metadata`, which reports the
    *installed distribution*, not the code being executed. Under an editable install —
    which is how the platform runs this package — those drift the moment the source tree
    moves ahead of the last `pip install`. Register **C-25** records a measured instance:
    with the tree at 1.0.0 and the dist-info still at 0.5.0, emitted frames were stamped
    `0.5.0` by 1.0.0 code, silently, in the artifact whose entire purpose is to be the
    record. A bare version number cannot distinguish the two states; a version plus a SHA
    can.

    A wheel install has no worktree and correctly stamps a bare version — that is a
    property of how the package was installed, not a failure, so no SHA is not an error
    (ADR-015's fault-versus-data-property test).

    Returns None if the distribution metadata cannot be resolved at all, e.g. running
    from an uninstalled source tree.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("views_evaluation")
    except PackageNotFoundError:
        return None
    sha = _source_git_sha()
    return f"{installed}+g{sha}" if sha else installed


@dataclass(frozen=True)
class MetricFrameMetadata:
    """
    Provenance for a MetricFrame, split by concern per ADR-020.

    The generic provenance is the reused views-frames ``FrameMetadata`` (so the vocabulary
    has one authority and consumers cannot diverge on key names). The evaluation-specific
    fields stay here and must never leak into ``FrameMetadata`` (register C-47).
    """

    provenance: FrameMetadata = field(default_factory=FrameMetadata)
    scoring_code_version: Optional[str] = None
    evaluation_timestamp: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Flatten to a plain dict (generic fields omit None; schema_version always present)."""
        d: Dict[str, Any] = dict(self.provenance.to_dict())
        if self.scoring_code_version is not None:
            d["scoring_code_version"] = self.scoring_code_version
        if self.evaluation_timestamp is not None:
            d["evaluation_timestamp"] = self.evaluation_timestamp
        d["schema_version"] = self.schema_version
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricFrameMetadata":
        """Reconstruct, routing generic keys to FrameMetadata and eval-specific keys here."""
        return cls(
            provenance=FrameMetadata.from_dict(data),
            scoring_code_version=data.get("scoring_code_version"),
            evaluation_timestamp=data.get("evaluation_timestamp"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


class MetricFrame:
    """
    A typed, transportable container of evaluation metric values.

    Attributes:
        values: float32 array of shape (N, 1) — one metric value per row, explicit trailing axis.
                NaN is permitted (a metric that was "not calculated" for that group).
        identifiers: dict mapping each axis name in AXES to a length-N string array.
        metadata: MetricFrameMetadata carrying generic + eval-specific provenance.
    """

    def __init__(
        self,
        values: np.ndarray,
        identifiers: Dict[str, np.ndarray],
        metadata: Optional[MetricFrameMetadata] = None,
    ):
        self._validate(values, identifiers)
        self.values = values
        self.identifiers = identifiers
        self.metadata = metadata or MetricFrameMetadata()

    @staticmethod
    def _validate(values: np.ndarray, identifiers: Dict[str, np.ndarray]) -> None:
        # ADR-013 fail-loud: structural envelope guarantees, checked at construction.
        if not isinstance(values, np.ndarray):
            err_msg = f"values must be a numpy array, got {type(values).__name__}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        if values.dtype != np.float32:
            err_msg = f"values must be float32, got {values.dtype}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        if values.ndim != 2:
            err_msg = (
                f"values must be 2D (N, 1) with an explicit trailing axis, got {values.ndim}D "
                f"with shape {values.shape}"
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        n_rows = values.shape[0]
        missing = set(AXES) - set(identifiers.keys())
        if missing:
            err_msg = (
                f"MetricFrame identifiers missing required axes: {sorted(missing)}. "
                f"Required: {list(AXES)}"
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        for key in AXES:
            arr = identifiers[key]
            if getattr(arr, "ndim", None) != 1:
                err_msg = (
                    f"Identifier '{key}' must be a 1D array, got "
                    f"{getattr(arr, 'ndim', '?')}D with shape {getattr(arr, 'shape', '?')}"
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
            if len(arr) != n_rows:
                err_msg = (
                    f"Identifier '{key}' length ({len(arr)}) mismatch values rows ({n_rows})"
                )
                logger.error(err_msg)
                raise ValueError(err_msg)

    @property
    def n_rows(self) -> int:
        return self.values.shape[0]

    # ── Serialization (own format — views-frames IO is (time, unit)-only) ───────
    def save(self, directory: Union[str, Path]) -> None:
        """Write values (.npy), identifiers (.npz), and metadata (.json) into ``directory``."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "values.npy", self.values)
        np.savez(path / "identifiers.npz", **self.identifiers)
        (path / "metadata.json").write_text(
            json.dumps(self.metadata.to_dict(), default=_json_default)
        )

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "MetricFrame":
        """Reconstruct a MetricFrame written by :meth:`save`."""
        path = Path(directory)
        values = np.load(path / "values.npy")
        with np.load(path / "identifiers.npz", allow_pickle=False) as idz:
            identifiers = {k: idz[k] for k in idz.files}
        metadata = MetricFrameMetadata.from_dict(json.loads((path / "metadata.json").read_text()))
        return cls(values=values, identifiers=identifiers, metadata=metadata)

    def __repr__(self) -> str:
        return (
            f"MetricFrame(n_rows={self.n_rows}, "
            f"axes={list(AXES)}, "
            f"schema_version='{self.metadata.schema_version}')"
        )
