param(
    [switch]$WithTestTools
)

$ErrorActionPreference = "Stop"

uv python install 3.11
if ($WithTestTools) {
    uv sync --locked --extra test
} else {
    uv sync --locked --extra api --extra app
}

Write-Host "AlphaForge is ready. Run .\\scripts\\check.ps1 to verify the research environment."
