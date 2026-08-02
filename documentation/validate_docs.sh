#!/usr/bin/env bash
# Validates internal consistency of base_docs documentation set.
# Exit 0 if clean, exit 1 if issues found.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

errors=0

echo "=== base_docs validation ==="
echo ""

# 1. Check for unfilled template placeholders in accepted/active files
#    (skip files whose names contain "template" — those are expected to have placeholders)
#    These are warnings only (non-blocking) since in the template repo some
#    files are legitimately Accepted with placeholder dates.
echo "--- Checking for template placeholders in accepted/active files ---"
warnings=0
while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    [[ "$file" == *template* ]] && continue
    if grep -q 'YYYY-MM-DD' "$file"; then
        echo "  WARN: Unfilled date placeholder in $file"
        warnings=$((warnings + 1))
    fi
    if grep -q '<roles / team>' "$file"; then
        echo "  WARN: Unfilled deciders placeholder in $file"
        warnings=$((warnings + 1))
    fi
    if grep -q '<ClassName>' "$file"; then
        echo "  WARN: Unfilled ClassName placeholder in $file"
        warnings=$((warnings + 1))
    fi
done < <(grep -rl 'Status:.*\(Accepted\|Active\)' --include='*.md' . 2>/dev/null || true)
if [ "$warnings" -eq 0 ]; then
    echo "  OK"
fi

# 2. Verify CIC active contracts exist (skip blockquote/example lines)
echo "--- Checking CIC active contract references ---"
if [ -f "CICs/README.md" ]; then
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        contract=$(echo "$line" | sed -n 's/^- `\(.*\.md\)`.*$/\1/p')
        if [ -n "$contract" ] && [ ! -f "CICs/$contract" ]; then
            echo "  ERROR: CIC contract listed but missing: CICs/$contract"
            errors=$((errors + 1))
        fi
    done < <(grep -E '^- `[A-Z].*\.md`' CICs/README.md 2>/dev/null | grep -v '>' || true)
fi

# 3. Cross-ADR reference integrity — ALL local ADRs (000-999), not just the
#    constitutional 00x band. Widened 2026-08-02: the previous 00x-only scope meant
#    a reference to any project ADR (010+) was unverified, so ADR-042 sat unindexed
#    and ADR-015's references went unchecked. See register C-34.
#
#    ADRs owned by OTHER repos are referenced here deliberately (e.g. views-frames
#    ADR-020 is the MetricFrame contract home) and must not be flagged, so a
#    reference qualified by a repo name is skipped.
echo "--- Checking cross-ADR references (all local ADRs) ---"
while IFS= read -r ref; do
    [[ -z "$ref" ]] && continue
    file=$(echo "$ref" | cut -d: -f1)
    line=$(echo "$ref" | cut -d: -f3-)
    # STRIP cross-repo references ("views-frames ADR-020", "views-reporting ADR-029")
    # rather than skipping the whole line. Skipping swallowed any local reference that
    # shared a line with a cross-repo one — a silent blind spot in a checker whose job
    # is finding blind spots (found by review-diff, 2026-08-02).
    line=$(echo "$line" | sed -E 's/(views-[a-z-]+|pipeline-core)[[:space:]]+ADR-[0-9]{3}//g')
    for adr_num in $(echo "$line" | grep -oP 'ADR-\K[0-9]{3}'); do
        match_count=$(find ADRs -name "${adr_num}_*.md" 2>/dev/null | wc -l)
        if [ "$match_count" -eq 0 ]; then
            echo "  ERROR: $file references ADR-${adr_num} but no matching file found"
            errors=$((errors + 1))
        fi
    done
done < <(grep -rn 'ADR-[0-9][0-9][0-9]' --include='*.md' . 2>/dev/null || true)

# 3b. Every ADR on disk must be indexed in ADRs/README.md
echo "--- Checking ADR index completeness ---"
for adr in ADRs/[0-9]*.md; do
    name=$(basename "$adr")
    [[ "$name" == *template* ]] && continue
    if ! grep -q "$name" ADRs/README.md 2>/dev/null; then
        echo "  ERROR: $name exists but is not indexed in ADRs/README.md"
        errors=$((errors + 1))
    fi
done

# 4. Check that referenced protocol files exist
echo "--- Checking protocol file references ---"
while IFS= read -r ref; do
    [[ -z "$ref" ]] && continue
    file=$(echo "$ref" | cut -d: -f1)
    proto=$(echo "$ref" | grep -oP 'contributor_protocols/[a-z_]+\.md' | head -1)
    if [ -n "$proto" ] && [ ! -f "$proto" ]; then
        echo "  ERROR: $file references $proto but file does not exist"
        errors=$((errors + 1))
    fi
done < <(grep -rn 'contributor_protocols/' --include='*.md' . 2>/dev/null || true)

# 5. Report template status markers
echo "--- Checking template status markers ---"
template_count=$(grep -rl '\-\-template\-\-' --include='*.md' . 2>/dev/null | wc -l)
echo "  INFO: $template_count files still have --template-- status (expected in template repo)"

echo ""
if [ "$errors" -gt 0 ]; then
    echo "=== FAILED: $errors issue(s) found ==="
    exit 1
else
    echo "=== PASSED: no issues found ==="
    exit 0
fi
