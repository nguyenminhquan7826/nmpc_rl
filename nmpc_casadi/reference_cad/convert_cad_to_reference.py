
from pathlib import Path

import ezdxf
import numpy as np
import pandas as pd
from ezdxf.path import make_path



BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "generated"
DS = 0.05                  # Khoang cach lay mau (m)
SPEED = 0.10               # Toc do tham chieu (m/s)
FLATTEN_MM = 1.0           # Sai so lay mau hinh hoc (mm)
WHEELBASE = 0.2515
STEERING_LIMIT = 0.55


def resample_by_arclength(points, ds):
    """Lay mau deu theo chieu dai cung."""
    points = np.asarray(points, dtype=float)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.r_[True, distances > 1e-9]
    points = points[keep]

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(segment_lengths)]

    if len(s) < 3 or s[-1] <= 0:
        raise ValueError("Quy dao khong du diem hoac co chieu dai bang 0.")

    s_new = np.arange(0.0, s[-1], ds)
    if len(s_new) == 0 or s_new[-1] < s[-1]:
        s_new = np.r_[s_new, s[-1]]

    x = np.interp(s_new, s, points[:, 0])
    y = np.interp(s_new, s, points[:, 1])
    return s_new, np.column_stack((x, y))


def normalize_path(points):
    """Dua diem dau ve (0,0), tiep tuyen dau theo truc +X."""
    points = np.asarray(points, dtype=float)

    # Bo diem trung lap
    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.r_[True, d > 1e-9]]

    if len(points) < 3:
        raise ValueError("Khong du diem de xac dinh tiep tuyen.")

    tangent = points[1] - points[0]
    angle = np.arctan2(tangent[1], tangent[0])

    c, s = np.cos(-angle), np.sin(-angle)
    rotation = np.array([[c, -s], [s, c]])

    return (points - points[0]) @ rotation.T


def calculate_reference(points):
    """Tinh cac dai luong tham chieu dong hoc."""
    s, xy = resample_by_arclength(points, DS)

    # Dao ham toa do theo chieu dai cung
    dx = np.gradient(xy[:, 0], s, edge_order=2)
    dy = np.gradient(xy[:, 1], s, edge_order=2)

    psi = np.unwrap(np.arctan2(dy, dx))
    dpsi = np.gradient(psi, s, edge_order=2)
    kappa = dpsi

    # Lam sach nhe do cong bang loc trung vi
    if len(kappa) >= 5:
        kappa = (
            kappa
            + np.roll(kappa, 1)
            + np.roll(kappa, -1)
        ) / 3.0
        kappa[0] = dpsi[0]
        kappa[-1] = dpsi[-1]

    steering = np.arctan(WHEELBASE * kappa)
    max_steering = float(np.max(np.abs(steering)))

    if max_steering > STEERING_LIMIT:
        raise ValueError(
            f"Goc lai yeu cau {max_steering:.4f} rad "
            f"vuot gioi han {STEERING_LIMIT:.4f} rad."
        )

    vx = np.full(len(s), SPEED)
    vy = np.zeros(len(s))
    r = vx * kappa
    t = s / SPEED

    return pd.DataFrame({
        "index": np.arange(len(s)),
        "s_m": s,
        "t_ref_s": t,
        "vx_ref_mps": vx,
        "vy_ref_mps": vy,
        "psi_ref_rad": psi,
        "r_ref_radps": r,
        "X_ref_m": xy[:, 0],
        "Y_ref_m": xy[:, 1],
        "kappa_1_per_m": kappa,
    })


def convert_spiral():
    filename = BASE / "QuyDao_Vong_Xoan.dxf"
    doc = ezdxf.readfile(filename)
    helix = next(e for e in doc.modelspace() if e.dxftype() == "HELIX")

    # Flattening giu thu tu tu dau den cuoi cua HELIX.
    # Dao nguoc de xe di tu vong trong ra vong ngoai.
    raw = np.array([
        [p.x, p.y] for p in helix.flattening(FLATTEN_MM)
    ])
    raw = raw[::-1]

    # DXF cua nguoi dung dung mm, chuyen sang m.
    points = raw / 1000.0
    points = normalize_path(points)

    return calculate_reference(points)


def convert_giun():
    filename = BASE / "Quy_Dao_giun1.dxf"
    doc = ezdxf.readfile(filename)
    entity = next(
        e for e in doc.modelspace()
        if e.dxftype() == "LWPOLYLINE"
    )

    # make_path giu thu tu dinh va bulge cua LWPOLYLINE.
    path = make_path(entity)
    raw = np.array([
        [p.x, p.y] for p in path.flattening(FLATTEN_MM)
    ])

    points = raw / 1000.0
    points = normalize_path(points)

    return calculate_reference(points)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("spiral_xref.csv", convert_spiral),
        ("giun_xref.csv", convert_giun),
    ]

    for filename, converter in tasks:
        df = converter()
        output_file = OUTPUT / filename
        df.to_csv(output_file, index=False, float_format="%.10f")

        print(f"\nDa tao: {output_file}")
        print(f"So diem: {len(df)}")
        print(f"Chieu dai: {df['s_m'].iloc[-1]:.3f} m")
        print(
            "Do cong lon nhat: "
            f"{df['kappa_1_per_m'].abs().max():.4f} 1/m"
        )
        print(
            "Goc lai lon nhat: "
            f"{np.arctan(WHEELBASE * df['kappa_1_per_m']).abs().max():.4f} rad"
        )
        print(
            "Diem dau: "
            f"({df['X_ref_m'].iloc[0]:.4f}, "
            f"{df['Y_ref_m'].iloc[0]:.4f}) m"
        )


if __name__ == "__main__":
    main()
