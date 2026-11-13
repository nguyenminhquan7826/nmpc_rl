"""Evaluate labelled turns from telemetry_rx CSV; requires Python standard library.
CSV yaw is already unwrapped radians. Gyro multiplier applies to CSV r only.
Reference angles must be independently measured, signed positive for left turn.
"""
import argparse,bisect,csv,json,math
from pathlib import Path

def interpolate(t,v,x):
 i=bisect.bisect_right(t,x)
 if i==0:return v[0]
 if i==len(t):return v[-1]
 return v[i-1]+(v[i]-v[i-1])*(x-t[i-1])/(t[i]-t[i-1])
def measure(t,yaw,rate,window,bias_dps):
 start=float(window['start_s']);end=float(window['end_s']);ref=window.get('reference_deg')
 if not math.isfinite(start+end+bias_dps) or not t[0]<=start<end<=t[-1]+1e-9:raise ValueError('Turn window outside recording or invalid')
 end=min(end,t[-1])
 if ref is not None:
  ref=float(ref)
  if not math.isfinite(ref):raise ValueError('Reference must be finite')
 knots=[start]+[x for x in t if start<x<end]+[end]
 values=[interpolate(t,rate,x) for x in knots]
 integral=math.degrees(sum((b-a)*(x+y)*.5 for a,b,x,y in zip(knots,knots[1:],values,values[1:])))
 heading=math.degrees(interpolate(t,yaw,end)-interpolate(t,yaw,start))
 corrected=integral-bias_dps*(end-start)
 return {'name':window.get('name','turn'),'start_s':start,'end_s':end,'duration_s':end-start,'reference_deg':ref,
  'heading_delta_deg':heading,'gyro_integral_deg':integral,'gyro_bias_corrected_integral_deg':corrected,
  'applied_bias_dps':bias_dps,'heading_error_vs_reference_deg':None if ref is None else heading-ref,
  'gyro_error_vs_reference_deg':None if ref is None else integral-ref,
  'gyro_corrected_error_vs_reference_deg':None if ref is None else corrected-ref,
  'heading_over_gyro_ratio':heading/integral if abs(integral)>1 else None,
  'heading_minus_gyro_deg':heading-integral}
def read_csv(path,multiplier):
 rows=list(csv.DictReader(Path(path).open()))
 if len(rows)<2 or any(r['accepted']!='1' for r in rows):raise ValueError('Need continuous accepted-only recording; split rejected segments first')
 t=[0.]
 for a,b in zip(rows,rows[1:]):
  dt=((int(b['imu_ms'])-int(a['imu_ms']))&0xffffffff)/1000
  if not 0<dt<=.2:raise ValueError('IMU duplicate, reset or gap >200 ms')
  t.append(t[-1]+dt)
 yaw=[float(r['yaw_est_rad']) for r in rows];rate=[float(r['r_measured_radps'])*multiplier for r in rows]
 if not all(math.isfinite(x) for x in yaw+rate):raise ValueError('Nonfinite yaw/gyro')
 return rows,t,yaw,rate
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('csv');p.add_argument('--windows',required=True,help='JSON list of labelled windows, seconds relative to first IMU timestamp')
 p.add_argument('--gyro-multiplier',type=int,choices=(-1,1),required=True);p.add_argument('--bias-dps',type=float,default=0);p.add_argument('--output',required=True);a=p.parse_args()
 if Path(a.output).resolve() in [Path(a.csv).resolve(),Path(a.windows).resolve()]:raise ValueError('Output must differ from input files')
 rows,t,yaw,rate=read_csv(a.csv,a.gyro_multiplier);windows=json.loads(Path(a.windows).read_text())
 if not isinstance(windows,list) or not windows:raise ValueError('Windows must be a nonempty JSON list')
 result={'source':a.csv,'duration_s':t[-1],'gyro_multiplier_applied_to_csv':a.gyro_multiplier,'turns':[measure(t,yaw,rate,w,a.bias_dps) for w in windows],
 'calibration_hex_values':sorted(set(r['calib_hex'] for r in rows)),
 'limitations':['Angles in reference_deg must come from independent observation','Bias is supplied explicitly, never fitted from the moving turn','No gain is applied or inferred as calibrated','No host timestamps: cannot establish STM32 clock scale or wire latency']}
 Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__':main()
