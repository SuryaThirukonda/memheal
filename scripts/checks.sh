#!/usr/bin/env bash
# The checks every push must pass. Run by .githooks/pre-push locally and by CI
# (.github/workflows/ci.yml), so a push that passes here passes there.
#
#   scripts/checks.sh                 all 8 checks
#   scripts/checks.sh --fast          skip the test suite (check 8)
#   scripts/checks.sh --junit FILE    also write the test report as JUnit XML
#
# No check may reach the network or spend model credits: the model API is pointed at a
# closed local port with a dummy key (a stray call fails instead of costing money), the
# Hugging Face hub is offline, and Mem0 telemetry is off. These override nothing in .env:
# python-dotenv never replaces variables that are already set.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)"
export OPENAI_API_KEY=checks-no-model-calls
export OPENAI_BASE_URL=http://127.0.0.1:9
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MEM0_TELEMETRY=False

FAST=0
JUNIT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --fast) FAST=1 ;;
        --junit) JUNIT="$2"; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

FAILED=()
run() {
    local name="$1"; shift
    printf '==> %s\n' "$name"
    if "$@"; then
        printf '    ok\n'
    else
        printf '    FAILED\n'
        FAILED+=("$name")
    fi
}

no_secrets() {
    # .env holds the API key and must never be tracked; no tracked file may contain one.
    if git ls-files --error-unmatch .env >/dev/null 2>&1; then
        echo "    .env is tracked"; return 1
    fi
    if git grep -nIE 'sk-[A-Za-z0-9_-]{20,}' -- . ':!uv.lock'; then
        echo "    a tracked file contains what looks like an API key"; return 1
    fi
}

no_large_or_run_data() {
    # Run folders and databases stay local (data/ is ignored). 50 MB is where GitHub starts
    # warning; the largest result file (the per-record preservation table) is ~31 MB.
    local bad=0
    if git ls-files data | grep -vx 'data/.gitkeep'; then
        echo "    files under data/ are tracked"; bad=1
    fi
    while IFS= read -r f; do
        [ -f "$f" ] || continue
        if [ "$(stat -c %s "$f")" -gt $((50 * 1024 * 1024)) ]; then
            echo "    $f is over 50 MB"; bad=1
        fi
    done < <(git ls-files)
    return $bad
}

local_only_hosts() {
    # Every attacker destination is a reserved .example name, so nothing can resolve.
    uv run --frozen python - <<'PY'
from memheal.scenarios import ALL_VARIANTS
bad = sorted({f"{k}: {d}" for k, s in ALL_VARIANTS.items()
              for d in s.attacker_destinations
              if not d.split("/")[0].endswith(".example")})
print("\n".join(f"    {b}" for b in bad[:20]))
raise SystemExit(1 if bad else 0)
PY
}

compiles() {
    uv run --frozen python -m compileall -q -x '(^|/)(\.venv|data)/' . >/dev/null
}

lint() {
    uv run --frozen ruff check .
}

lockfile() {
    uv lock --check
}

clis_import() {
    # Every command-line entry point still imports (a broken import is the most common
    # way a refactor breaks a sweep that the unit tests never start).
    uv run --frozen python -c "
import importlib
for m in ('run_demo', 'recover', 'measure', 'diagnose', 'analyze', 'analyze_main',
          'analyze_history', 'bcp', 'cost', 'batch', 'build_external', 'chat'):
    importlib.import_module(m)
"
}

tests() {
    local extra=()
    [ -n "$JUNIT" ] && extra+=("--junitxml=$JUNIT")
    uv run --frozen pytest -q -p no:cacheprovider "${extra[@]}"
}

run "1/8 no secrets tracked (.env, API keys)"        no_secrets
run "2/8 no run data or files over 50 MB tracked"    no_large_or_run_data
run "3/8 attacker hosts are .example only"           local_only_hosts
run "4/8 all Python files compile"                   compiles
run "5/8 lint (ruff: syntax, undefined, unused)"     lint
run "6/8 uv.lock matches pyproject.toml"             lockfile
run "7/8 every CLI entry point imports"              clis_import
if [ "$FAST" = 1 ]; then
    printf '==> 8/8 offline test suite\n    skipped (--fast)\n'
else
    run "8/8 offline test suite (no network, no model calls)" tests
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    printf '\n%d check(s) failed:\n' "${#FAILED[@]}"
    printf '  - %s\n' "${FAILED[@]}"
    exit 1
fi
printf '\nall checks passed\n'
