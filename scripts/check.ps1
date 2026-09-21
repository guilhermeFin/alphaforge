param(
    [switch]$Coverage
)

$ErrorActionPreference = "Stop"

uv sync --locked --extra test
if ($Coverage) {
    uv run --locked --no-sync pytest --cov --cov-report=term-missing --cov-report=xml
} else {
    uv run --locked --no-sync pytest
}
