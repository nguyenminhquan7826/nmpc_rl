@echo off
cd /d "%~dp0"
python rl_nmpc.py --mode train --episodes 100 --seconds 120 --seed 0 --output rl_results\seed0
pause
