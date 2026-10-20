# Direct-P Actor NN + six-parameter Critic: held-out S-bend experiment

**Run:** https://github.com/nguyenminhquan7826/nmpc_rl/actions/runs/39417166327  
**Artifact:** https://github.com/nguyenminhquan7826/nmpc_rl/actions/runs/39417166327/artifacts/12262440003

## Protocol

- One independent model, seed 0, 20 training episodes, 120-s episode cap, original training CAD track.
- Actor: neural network from R3A architecture; LR=0.001, Gaussian exploration sigma=0.05.
- Critic: six directly updated positive diagonal P parameters; projected TD(0) semi-gradient LR=0.001; update per transition.
- Test: generated independent analytic `heldout_s_bend` with equal reference length but distinct geometry. **Not a second user-drawn AutoCAD file**.
- Test plant profiles: nominal, +10% payload ramp, -20% tire stiffness on segment, +25% rolling resistance on segment.
- Paired arms: fixed NMPC, trained Actor with *initial* P, trained Actor with *learned* P (frozen sampled), trained Actor/learned P (frozen deterministic mean), trained Actor/learned P with **online test-time updates**.
- Held-out test results were not used to choose the checkpoint. The **final** training checkpoint is evaluated, not a best-validation checkpoint.
- All held-out runs reached endpoint with 100% eventual solver success. Retry counts are separately reported in CSV.

## Unseen path CTE (cm; mean over post-1.3 m segment)

| Profile | Fixed NMPC | Actor NN + initial P | Actor NN + learned P (frozen) | Actor NN + learned P (online) |
|---|---:|---:|---:|---:|
| Nominal | 4.8185 | 6.4809 | 5.7365 | 5.5840 |
| Payload | 4.6931 | 6.2650 | 5.5640 | 5.4002 |
| Tire segment | 4.5298 | 5.9350 | 5.2606 | 5.1393 |
| Rolling segment | 5.3731 | 7.0748 | 6.2956 | 6.1110 |
| **Mean** | **4.8536** | **6.4389** | **5.7141** | **5.5586** |

Interpretation: relative to exactly the same trained Actor held frozen and initial P, learned P improves post-segment mean CTE by approx 11.26%. Compared to fixed NMPC, the trained Actor+learned P is approx 17.73% worse on held-out CTE. Online adaptation on the held-out path improves CTE slightly but **is not strict zero-shot generalization**.

## Heading tradeoff

Nominal held-out RMSE heading (rad):
- Fixed NMPC: 0.019124.
- Actor NN + initial P: 0.003805.
- Actor NN + learned P (frozen): 0.007424.
- Direct-P online: 0.010083.

Therefore learned Actor+P tracks heading more closely than fixed NMPC but produces worse CTE. This is a position/heading tradeoff, not evidence of uniformly superior tracking.

## Critical training caveat

Episodes **17 and 19** on the original training CAD route terminated with `solver_fail_3x`, not successful completion. Episode 19 endpoint:
- `solver_success_pct = 98.1716%`; `completion_pct = 70.6547%`; 13 failed steps.
- Training CTE after 1.3m: 9.5484 cm, with a catastrophic failure penalty.
- Physical terminal diagonal at final saved checkpoint approximately `[307.7144, 353.7546, 1068.6400, 120.5041, 4871.5772, 4951.2148]`, compared with initial `[240,60,1000,120,2015,2015]`.
- Parameter P kept drifting upward across 20 training episodes; **no convergence established**.
- The original runner overwrites `checkpoint.npz` each episode. Thus the final model used on held-out tests may not be the best model from training, and this test does **not** establish Direct-P's best achievable transfer performance.

## Scientific next step

Save each checkpoint, select using a predefined **training-geometry validation set** (e.g. different initial heading and lateral offsets), and freeze that checkpoint before evaluation on held-out geometry. Evaluate 3+ independent random seeds and compare to the original neural Critic on the same scenarios. Keep terminal TD(0) update unchanged during this protocol fix. Test-only updates must be marked online adaptation, not frozen generalization.

Files: `direct_results/heldout_train_seed0/checkpoint.npz`, `training_history.csv`, `direct_results/heldout_eval_seed0/comparison.csv`, per-controller per-profile traces, reference manifest, and metrics are all in the GitHub Actions artifact linked above.
