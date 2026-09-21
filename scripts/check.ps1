param(
    [switch]$Coverage
)

$ErrorActionPreference = "Stop"

uv sync --locked --extra test
if ($Coverage) {
    uv run --locked --no-sync pytest --cov --cov-report=term-missing --cov-report=xml --cov-report=json
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    uv run --locked --no-sync python scripts/check_coverage.py coverage.json
} else {
    uv run --locked --no-sync pytest
}
