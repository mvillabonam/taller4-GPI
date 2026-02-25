$ErrorActionPreference = "Stop"

Write-Host "=== PIPELINE AUTOMATIZADO: FINANCIAL SIMULATION ==="

Write-Host "1) Generando simulacion financiera"
py .\proyecto\scripts\generate_financial_simulation.py

Write-Host "2) Procesamiento de datos"
py .\proyecto\scripts\processing.py

Write-Host "3) Estadisticas descriptivas"
py .\proyecto\scripts\descriptives.py

Write-Host "4) Pronosticos"
py .\proyecto\scripts\forecasts.py

Write-Host "5) Modelos"
py .\proyecto\scripts\modelado.py

Write-Host "=== PIPELINE FINALIZADO ==="