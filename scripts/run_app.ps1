# Launch the AlphaForge API + Streamlit UI (run from anywhere)
$repo = Split-Path -Parent $PSScriptRoot
Write-Host "AlphaForge repo: $repo"
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "Set-Location '$repo'; uvicorn api.main:app --port 8000"
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "Set-Location '$repo'; streamlit run app/Home.py"
Write-Host "API   -> http://127.0.0.1:8000/docs"
Write-Host "UI    -> http://localhost:8501"
