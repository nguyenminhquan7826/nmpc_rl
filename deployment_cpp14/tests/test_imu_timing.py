"""Host-only audit validation; requires numpy, no hardware."""
import importlib.util,sys
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('audit',Path(__file__).parents[1]/'tools/audit_imu_timing.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
t=np.arange(0,12,.01)
yaw=.4*np.sin(t*2)+.1*np.sin(t*5)
# Measured gyro delayed by 80ms relative to known yaw derivative.
r=.8*np.cos((t-.08)*2)+.5*np.cos((t-.08)*5)
r0=.8*np.cos(t*2)+.5*np.cos(t*5)
assert abs(audit.fit(t,yaw,r)['best_unscaled']['lag_ms']-80)<=5
assert abs(audit.fit(t,yaw,r0)['best_unscaled']['lag_ms'])<=5
try:audit.fit(np.array([0,.3]*20),np.zeros(40),np.zeros(40));raise AssertionError('Accepted bad timestamps')
except ValueError:pass
print('Known 80ms delay, zero delay and invalid timestamp audit PASS')
