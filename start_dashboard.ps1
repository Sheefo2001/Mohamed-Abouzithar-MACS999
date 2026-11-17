$Root = $PSScriptRoot
if (-not $Root) { $Root = Split-Path -Parent $MyInvocation.MyCommand.Definition }
if (-not $Root) { $Root = (Get-Location).Path }

Start-Process powershell -ArgumentList "-NoExit", "-Command", "uvicorn src.api.main:app --reload --port 8000" -WorkingDirectory $Root
Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm run dev" -WorkingDirectory (Join-Path $Root "webapp")
