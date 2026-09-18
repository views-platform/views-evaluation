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
import re
import tomllib
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


# PEP 503-normalised name of this distribution; the ownership gate below compares against it.
_DISTRIBUTION_NAME = "views-evaluation"


def _nested_str(data: Any, *keys: str) -> Optional[str]:
    """``data[k1][k2]...`` if every level is a table and the leaf a string, else None.

    A pyproject with ``project = "x"`` or ``[tool] poetry = 1`` is not this
    distribution's; it must read as "not ours" rather than raise AttributeError.
    """
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data if isinstance(data, str) else None


def _normalised_project_name(pyproject: Path) -> Optional[str]:
    """The distribution name a pyproject.toml declares, PEP 503-normalised, or None.

    Parsed with ``tomllib`` (stdlib at this package's Python floor) rather than matched
    as text, so ``name = "views_evaluation"`` inside an unrelated table (an import-linter
    contract, a dependency line, a description) is not mistaken for ownership.
    """
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))  # TOML is UTF-8 by spec
    name = _nested_str(data, "project", "name") or _nested_str(data, "tool", "poetry", "name")
    if name is None:
        return None
    return re.sub(r"[-_.]+", "-", name).lower()


def _read_head_sha(git: Path) -> str:
    """The commit HEAD points at, as a 7-char prefix. Raises when it cannot be read.

    Handles a ``.git`` directory, a ``.git`` file pointing at a linked worktree's or a
    submodule's real dir, a symbolic HEAD with a loose or packed ref, and a linked
    worktree's ``commondir`` (refs and ``packed-refs`` live in the main repository's dir,
    HEAD in the worktree's own). Every read is UTF-8. A symbolic-ref chain is not
    followed, an unborn branch has no commit, and the reftable ref storage (git 2.45+,
    ``--ref-format=reftable``) is not parsed: all three raise rather than return a
    non-hash, so the caller can log them.
    """
    if git.is_file():  # linked worktree or submodule: `.git` is a file pointing at the real dir
        gitdir = git.read_text(encoding="utf-8").split("gitdir:", 1)[1].strip()
        git = (git.parent / gitdir).resolve()  # relative to the file, never to the cwd
    commondir = git / "commondir"
    refs_root = (git / commondir.read_text(encoding="utf-8").strip()).resolve() if commondir.exists() else git
    if (git / "reftable").is_dir() or (refs_root / "reftable").is_dir():
        # git >= 2.45 `--ref-format=reftable`: refs live in a binary table, not in
        # `refs/` or `packed-refs`. Checked in both the worktree's own dir and the
        # common dir, before HEAD is read, so the format is named whatever else is
        # missing. Not read here; say so instead of "unborn branch".
        raise ValueError("refs are stored in git's reftable format, which is not read")
    head = (git / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref:"):
        ref = head.split(":", 1)[1].strip()
        target = refs_root / ref
        value = None
        if target.exists():
            value = target.read_text(encoding="utf-8").strip()
        else:
            packed = refs_root / "packed-refs"
            if packed.exists():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line.endswith(f" {ref}"):
                        value = line.split()[0]
                        break
        if value is None:
            raise ValueError(f"{ref} has no commit (unborn branch, or ref not found)")
    else:
        value = head
    if not re.fullmatch(r"[0-9a-f]{7,64}", value):
        raise ValueError(f"HEAD resolved to {value[:24]!r}, which is not a commit hash")
    return value[:7]


def _source_git_sha() -> Optional[str]:
    """Short SHA of THIS repository's checkout when the module runs from it, else None.

    The check is bounded to one directory: the one containing the ``views_evaluation/``
    package directory (``parents[2]`` of this file under the repository's flat layout —
    a ``src/`` layout would need ``parents[3]``, and until it got it every editable
    install would stamp bare; the tests anchor their fixtures to the module's path
    relative to the repository root so that move turns them red). A ``.git`` directory
    or ``.git`` file there counts only if a ``pyproject.toml`` beside it declares this
    distribution. No parent of that directory is consulted for ``.git``; a ``.git``
    *file* is followed to wherever it points, which is how git lays out worktrees and
    submodules.

    Why bounded: the first version walked every parent to the nearest ``.git``. A wheel
    installed into a virtualenv nested inside a consumer's checkout (uv's default
    ``.venv``) sits below the *consumer's* ``.git``, and the walk stamped the consumer's
    commit as this library's version, in the artifact whose purpose is to be the record
    (register C-39). A wheel install must stamp a bare version wherever it lives.

    Two ``None`` outcomes are contracted data properties and log nothing: no ``.git`` or
    no ``pyproject.toml`` at that directory (a wheel), or a ``pyproject.toml`` naming
    another distribution (a vendored copy). A third is a fault and logs at WARNING: a
    ``.git`` is present for this distribution but no commit could be read from it
    (unreadable file, corrupt pointer, symbolic-ref chain, unborn branch, undecodable
    bytes, reftable ref storage). The stamp is still the bare version — the caller may pass
    ``scoring_code_version`` explicitly — but the trace is left (logging standard §5.1).

    Read from ``.git`` directly rather than shelling out: ``git`` is not guaranteed to
    exist where evaluations run, and a missing binary must not change what gets stamped.
    """
    try:
        root = Path(__file__).resolve().parents[2]
    except IndexError:
        return None
    git = root / ".git"
    pyproject = root / "pyproject.toml"
    if not git.exists() or not pyproject.exists():
        return None
    try:
        if _normalised_project_name(pyproject) != _DISTRIBUTION_NAME:
            return None
        return _read_head_sha(git)
    except (OSError, ValueError, IndexError, RuntimeError) as exc:
        # ValueError covers tomllib.TOMLDecodeError and UnicodeDecodeError; IndexError a
        # `.git` file without `gitdir:`; RuntimeError is pathlib's symlink-loop signal on
        # Python 3.11/3.12 (3.13+ raises OSError for the same tree).
        logger.warning(
            "scoring_code_version: %s exists but no commit could be read from it "
            "(%s: %s); stamping the bare version", git, type(exc).__name__, exc,
        )
        return None


def default_scoring_code_version() -> Optional[str]:
    """The version stamped into an emitted MetricFrame's provenance.

    The installed distribution's version, plus `+g<sha>` when this module is running
    from this repository's own checkout — a `.git` beside the `views_evaluation/`
    package directory, with a `pyproject.toml` there naming this distribution (see
    `_source_git_sha`). A wheel built from an untagged commit carries no SHA and stamps
    the version it was built as, indistinguishable from the tagged release.

    **Why the SHA.** The version alone comes from `importlib.metadata`, which reports the
    *installed distribution*, not the code being executed. Under an editable install —
    which is how the platform runs this package — those drift the moment the source tree
    moves ahead of the last `pip install`. Register **C-25** records a measured instance:
    with the tree at 1.0.0 and the dist-info still at 0.5.0, emitted frames were stamped
    `0.5.0` by 1.0.0 code, silently, in the artifact whose entire purpose is to be the
    record. A bare version number cannot distinguish the two states; a version plus a SHA
    can.

    A wheel install stamps a bare version wherever it lives, including inside another
    repository's checkout (register C-39). That is a property of how the package was
    installed, not a failure, so no SHA is not an error (ADR-015's fault-versus-data-
    property test).

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
