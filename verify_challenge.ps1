# verify_challenge.ps1

Write-Host "--- Step 1: generating Data ---" -ForegroundColor Cyan
& .\venv\Scripts\python data/generate_data.py

Write-Host "`n--- Step 2: Running Random Forest Baseline ---" -ForegroundColor Cyan
Push-Location starter_code
& ..\venv\Scripts\python baseline.py
Pop-Location

Write-Host "`n--- Step 3: Running GNN Baseline ---" -ForegroundColor Cyan
Push-Location starter_code
& ..\venv\Scripts\python baseline_gnn.py
Pop-Location

Write-Host "`n--- Step 4: Scoring Submissions ---" -ForegroundColor Cyan
Write-Host "1. Random Forest Score:" -NoNewline
& .\venv\Scripts\python scoring_script.py submissions/sample_submission.csv

Write-Host "2. GNN Score:" -NoNewline
& .\venv\Scripts\python scoring_script.py submissions/gnn_submission.csv
