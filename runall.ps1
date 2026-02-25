$ErrorActionPreference = "Stop"

Write-Host "=== PIPELINE AUTOMATIZADO: FINANCIAL SIMULATION ==="

Write-Host "1) Generando simulacion financiera"
py .\scripts\generate_financial_simulation.py

Write-Host "2) Procesamiento de datos"
py .\scripts\processing.py

Write-Host "3) Estadisticas descriptivas"
py .\scripts\descriptives.py

Write-Host "4) Pronosticos"
py .\scripts\forecasts.py

Write-Host "=== PIPELINE FINALIZADO ==="
