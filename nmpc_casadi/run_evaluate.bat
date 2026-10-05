@echo off
cd /d "%~dp0"
python rl_nmpc.py --mode evaluate --seconds 120 --checkpoint rl_results\seed0\checkpoint.npz --output rl_results\evaluation_seed0
pause
