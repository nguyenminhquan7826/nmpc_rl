@echo off
cd /d "%~dp0"
python rl_nmpc.py --mode evaluate --checkpoint results_v8_critic_n16\checkpoint_best_tracking.npz --output eval_v8_critic_n16_tracking
pause
