# TEST reference from the second waypoint map

`test_xref.csv` is generated from `source_waypoints.csv`, originally uploaded
as `reference_waypoints(1).csv`. It differs from the pinned TRAIN track.
The earlier uploaded file named test_xref.csv was identical to TRAIN and
must not be used as an independent test route.

The generator uses natural cubic splines, numerical arc-length
reparameterization, tangent heading, geometric curvature, and a
curvature-limited speed profile with forward/backward acceleration passes.
The reference has 430 points, length about 10.704 m and duration 111.62 s.
Speed ranges from 0.0663 to 0.10 m/s. There is no endpoint stop.

Experimental generator settings (not measured vehicle limits): lateral
acceleration cap 0.01 m/s², acceleration/braking cap 0.05 m/s². See
reference_config.json and test_reference.png. Geometry/kinematics and
repository input validation passed; full vehicle tracking is not verified.
Spline geometry is interpolated, not identical to every original XY sample.

From nmpc_casadi, evaluate without retraining (adjust checkpoint path):

```powershell
python evaluate_controller_modes.py --checkpoint results_v14/checkpoint_best_frozen.npz --reference test_reference/test_xref.csv --output evaluation_v14_test --max-steps 2500 --seeds 0 1 2 --sampling-control
```

Adaptation modes learn on this route; keep their scores separate from frozen
test scores. Repeated tuning on this route makes it a validation route.

To regenerate, from this folder (existing generated outputs are replaced):

```powershell
python -m pip install -r requirements-generation.txt
python generate_reference.py
```

The pre-generated CSV needs no SciPy at evaluation time. TRAIN files and
reference_split are not changed. This folder is versioned because the
reference_split folder is ignored by this repository.
