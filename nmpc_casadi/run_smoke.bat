@echo off
cd /d "%~dp0"
python rl_nmpc.py --mode smoke --seconds 3 --output rl_results\smoke
pause
