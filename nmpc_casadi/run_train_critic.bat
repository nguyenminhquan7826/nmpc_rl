@echo off
cd /d "%~dp0"
python rl_nmpc.py --mode train --train-component critic --episodes 100 --critic-lr 0.003 --critic-nsteps 16 --rollout-steps 256 --eval-every 10 --seed 0 --initial-seed 0 --output results_v8_critic_n16
pause
