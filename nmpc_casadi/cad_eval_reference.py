"""Read a user-supplied AutoCAD XREF for EVALUATION ONLY.

Training reference files are never modified. The supplied source CSV is copied
unchanged into the evaluation output directory and receives a horizon-only
virtual tail that preserves its longitudinal speed profile.
"""
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from hashlib import sha256


def sha256_file(path):
    """Checksum reference bytes, independently of the removed RL evaluation tools."""
    digest = sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


REQUIRED_XREF_COLUMNS = (
    'index', 's_m', 't_ref_s', 'vx_ref_mps', 'vy_ref_mps',
    'psi_ref_rad', 'r_ref_radps', 'X_ref_m', 'Y_ref_m',
    'kappa_1_per_m',
)


def prepare_cad_evaluation_reference(
    source_csv, output_dir, tail_length_m,
    *, min_speed_mps=0.05, max_speed_mps=0.10,
    wheelbase_m=0.2515, steering_limit_rad=0.55,
):
    """Validate and stage a held-out XREF without changing the training track.

    The source is an open-path XREF with the project's 10 reference columns.
    Progress s is in meters; speed is in m/s. The physical reference is
    unchanged. A separate virtual horizon tail uses final reference speed.
    """
    src = Path(source_csv).expanduser().resolve(strict=True)
    out_dir = Path(output_dir).expanduser().resolve()
    if src.suffix.lower() != '.csv':
        raise ValueError('Evaluation XREF must be a CSV file')
    if src.parent == out_dir:
        raise ValueError('Keep the supplied CAD XREF outside the evaluation output directory')
    if not np.isfinite(tail_length_m) or tail_length_m <= 0:
        raise ValueError('Virtual tail length must be positive and finite')
    data = pd.read_csv(src)
    missing = [c for c in REQUIRED_XREF_COLUMNS if c not in data]
    if missing:
        raise ValueError(f'CAD XREF lacks required columns: {missing}')
    if len(data) < 3:
        raise ValueError('CAD XREF requires >=3 points')
    if not all(pd.api.types.is_numeric_dtype(data[c]) for c in REQUIRED_XREF_COLUMNS):
        raise ValueError('All CAD XREF columns must be numeric')
    vals = data[list(REQUIRED_XREF_COLUMNS)].to_numpy(dtype=float)
    if not np.all(np.isfinite(vals)):
        raise ValueError('CAD XREF contains NaN or Inf')
    s = data['s_m'].to_numpy(float)
    ds = np.diff(s)
    if np.any(ds <= 0):
        raise ValueError('CAD arc length s_m must increase strictly')
    if abs(s[0]) > 1e-7:
        raise ValueError('CAD XREF must begin at s_m=0')
    vx = data['vx_ref_mps'].to_numpy(float)
    vy = data['vy_ref_mps'].to_numpy(float)
    # The present XREF generator assumes zero lateral reference speed.
    if np.any(np.abs(vy) > 1e-8):
        raise ValueError('Current CAD evaluator requires vy_ref_mps=0')
    if not (0 < min_speed_mps <= max_speed_mps):
        raise ValueError('Inconsistent minimum and maximum test speed limits')
    if np.any(vx < min_speed_mps - 1e-9) or np.any(vx > max_speed_mps + 1e-9):
        raise ValueError(
            f'CAD vx_ref outside NMPC/test range [{min_speed_mps}, {max_speed_mps}] m/s'
        )
    kappa = data['kappa_1_per_m'].to_numpy(float)
    yaw = data['r_ref_radps'].to_numpy(float)
    if not np.allclose(yaw, vx * kappa, atol=1e-4, rtol=1e-4):
        raise ValueError('CAD yaw-rate reference does not match vx_ref * curvature')
    if np.max(np.abs(np.arctan(wheelbase_m*kappa))) > steering_limit_rad + 1e-9:
        raise ValueError('CAD reference requires steering beyond NMPC steering limits')
    xy = data[['X_ref_m', 'Y_ref_m']].to_numpy(float)
    chord = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if np.any(chord <= 1e-9) or np.any(chord > ds * 1.05):
        raise ValueError('CAD XY segments inconsistent with arc-length s_m')
    expected_t = np.r_[0., np.cumsum(ds / (0.5*(vx[:-1] + vx[1:])))]
    if not np.allclose(
        data['t_ref_s'].to_numpy(float) - float(data['t_ref_s'].iloc[0]),
        expected_t, rtol=1e-5, atol=1e-5,
    ):
        raise ValueError('CAD reference t_ref_s inconsistent with vx_ref and arc length')

    staged = out_dir / 'cad_eval_source_xref.csv'
    tail_file = out_dir / 'cad_eval_with_virtual_tail.csv'
    if staged.exists() or tail_file.exists():
        raise FileExistsError('CAD evaluation staging paths already exist')
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, staged)
    if sha256_file(staged) != sha256_file(src):
        raise RuntimeError('Staged evaluation XREF differs from original')

    median_ds = float(np.median(ds))
    n_tail = max(2, int(np.ceil(float(tail_length_m)/median_ds)))
    step = float(tail_length_m) / n_tail
    last = data.iloc[-1]
    x_now = float(last['X_ref_m'])
    y_now = float(last['Y_ref_m'])
    psi_now = float(last['psi_ref_rad'])
    s_now = float(last['s_m'])
    time_now = float(last['t_ref_s'])
    speed_end = float(vx[-1])
    curvature_end = float(kappa[-1])
    rows = []
    for j in range(1, n_tail + 1):
        curvature_j = curvature_end*(1. - j/n_tail)
        psi_mid = psi_now + .5*curvature_j*step
        x_now += step*np.cos(psi_mid)
        y_now += step*np.sin(psi_mid)
        psi_now += curvature_j*step
        s_now += step
        time_now += step/speed_end
        row = last.copy()
        row['index'] = int(last['index']) + j
        row['s_m'] = s_now
        row['t_ref_s'] = time_now
        row['vx_ref_mps'] = speed_end
        row['vy_ref_mps'] = 0.
        row['psi_ref_rad'] = psi_now
        row['r_ref_radps'] = speed_end*curvature_j
        row['X_ref_m'] = x_now
        row['Y_ref_m'] = y_now
        row['kappa_1_per_m'] = curvature_j
        rows.append(row)
    extended = pd.concat([data, pd.DataFrame(rows, columns=data.columns)], ignore_index=True)
    extended.to_csv(tail_file, index=False)
    return {
        'cad_s_end': float(s[-1]),
        'cad_X': xy[:, 0], 'cad_Y': xy[:, 1],
        'tail_s_end': float(s_now),
        'n_original': int(len(data)),
        'n_tail': n_tail,
    }, {
        'id': 'cad_eval_external',
        'held_out': True,
        'user_supplied_evaluation_track': str(src),
        'reference_file': str(staged),
        'reference_sha256': sha256_file(staged),
        'tail_file': str(tail_file),
        'tail_sha256': sha256_file(tail_file),
        'reference_length_m': float(s[-1]),
        'original_count': int(len(data)),
        'min_reference_speed_mps': float(vx.min()),
        'max_reference_speed_mps': float(vx.max()),
        'max_abs_curvature_1_per_m': float(np.max(np.abs(kappa))),
        'profile_preserved_from_source': True,
    }
