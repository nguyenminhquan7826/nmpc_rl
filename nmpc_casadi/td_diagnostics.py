"""Signed episode TD statistics, separate from task reward and RMS."""
import numpy as np


def episode_td_statistics(frame):
    td=frame["td"].to_numpy(dtype=float)
    if not len(td) or not np.isfinite(td).all():
        raise ValueError("Episode TD must be nonempty and finite")
    result={"mean_td":float(np.mean(td)),
            "rms_td":float(np.sqrt(np.mean(td**2))),
            "mean_abs_td":float(np.mean(np.abs(td))),
            "std_td":float(np.std(td)),
            "negative_td_pct":float(100*np.mean(td<0)),
            "positive_td_pct":float(100*np.mean(td>0))}
    for i in range(6):
        # Raw semi-gradient signal; projection may change the applied step.
        result[f"mean_td_e2_{i}"]=float(np.mean(td*frame[f"e_{i}"].to_numpy(float)**2))
    return result
