"""Estimate apparent STM32 tx clock rate versus host monotonic read time.
Not a physical latency/oscillator calibration: changing receive backlog can
mimic clock drift. Requires a fresh --device --host-timestamps CSV, not replay.
Python standard library only; never modifies source timestamps or live clocks.
"""
import argparse,csv,json,math,statistics
from pathlib import Path

def fit(points):
 # Bound the pairwise robust fit even for very long files.
 if len(points)>200:points=[points[round(k*(len(points)-1)/199)] for k in range(200)]
 slopes=[(b[1]-a[1])/(b[0]-a[0]) for i,a in enumerate(points) for b in points[i+1:] if b[0]>a[0]]
 if not slopes:raise ValueError('Not enough distinct host timestamps')
 slope=statistics.median(slopes)
 intercept=statistics.median(y-slope*x for x,y in points)
 residual=[y-(slope*x+intercept) for x,y in points]
 mx=statistics.mean(x for x,y in points);my=statistics.mean(y for x,y in points)
 denominator=sum((x-mx)**2 for x,y in points)
 ols=sum((x-mx)*(y-my) for x,y in points)/denominator
 return slope,ols,residual,len(points)
def analyze(rows,warmup_s=5.,bin_s=5.,min_duration_s=60.):
 if not all(math.isfinite(v) and v>0 for v in [bin_s,min_duration_s]) or not math.isfinite(warmup_s) or warmup_s<0:raise ValueError('Invalid analysis durations')
 required=['accepted','tx_ms','host_rx_monotonic_ns','host_rx_batch_id']
 if not rows or any(k not in rows[0] for k in required):raise ValueError('Missing host timestamps: record with --device --host-timestamps; old CSV cannot be reconstructed')
 accepted=[r for r in rows if r['accepted']=='1'];discarded=len(rows)-len(accepted)
 if len(accepted)<2:raise ValueError('Not enough accepted samples')
 groups=[];last_ns=None;last_id=None;last_tx=None;source_s=0.;origin=None;frames=0
 for r in accepted:
  ns=int(r['host_rx_monotonic_ns']);bid=int(r['host_rx_batch_id']);tx=int(r['tx_ms'])
  if ns<0 or bid<1 or not 0<=tx<2**32:raise ValueError('Invalid timestamp or batch ID')
  if origin is None:origin=ns
  if last_tx is not None:
   delta=(tx-last_tx)%2**32
   if delta==0 or delta>=2**31:raise ValueError('Duplicate, reordered or reset STM32 time; split recording')
   source_s+=delta/1000
   if ns<last_ns or bid<last_id:raise ValueError('Host time/batch ordering invalid')
   if bid==last_id and ns!=last_ns:raise ValueError('Same read batch must share one host timestamp')
  host_s=(ns-origin)*1e-9  # Subtract integer nanoseconds BEFORE converting to float.
  frames=frames+1 if bid==last_id else 1
  if groups and (bid==last_id or ns==last_ns):
   # Latest source packet in a batch reduces older-frame buffering influence.
   groups[-1]=(host_s,source_s)
  else:
   groups.append((host_s,source_s))
  rcount=frames
  if last_tx is None:max_frames=rcount
  else:max_frames=max(max_frames,rcount)
  last_ns=ns;last_id=bid;last_tx=tx
 duration=groups[-1][0]-groups[0][0]
 if duration<min_duration_s:raise ValueError(f'Host duration {duration:.2f}s is too short; record at least {min_duration_s:g}s, preferably 120s')
 bins={}
 for h,s in groups:
  if h<warmup_s:continue
  bins.setdefault(int((h-warmup_s)/bin_s),[]).append((h,s))
 points=[(statistics.median(h for h,s in v),statistics.median(s for h,s in v)) for k,v in sorted(bins.items()) if v]
 if len(points)<8:raise ValueError('Need at least 8 populated time bins after warmup')
 slope,ols,residual,count=fit(points)
 if not math.isfinite(slope) or slope<=0:raise ValueError('Nonpositive or invalid clock fit')
 half=len(points)//2
 halves=[fit(points[:half])[0],fit(points[half:])[0]] if half>=4 else None
 gaps=[b[0]-a[0] for a,b in zip(groups,groups[1:])]
 warnings=[]
 if max(gaps)>1.:warnings.append('Host read gap >1s: scheduler/transport buffering may affect fit')
 if halves and abs(halves[0]-halves[1])*1e6>1000:warnings.append('Half-recording slopes differ by >1000ppm; clock ratio may be confounded by buffering/nonstationarity')
 if discarded:warnings.append('Rejected odometry rows excluded; review source faults')
 return {'accepted_rows':len(accepted),'rejected_rows_excluded':discarded,'distinct_read_time_groups':len(groups),'max_frames_per_batch':max_frames,
  'host_elapsed_s':duration,'stm32_tx_elapsed_s':source_s,'warmup_excluded_s':warmup_s,'bin_width_s':bin_s,'fit_points':count,
  'apparent_stm32_seconds_per_host_second':slope,'apparent_drift_ppm':(slope-1)*1e6,'apparent_drift_percent':(slope-1)*100,
  'ols_seconds_per_host_second':ols,'first_second_half_ratios':halves,
  'binned_fit_residual_rms_ms':math.sqrt(statistics.mean(v*v for v in residual))*1000,
  'max_host_read_gap_s':max(gaps),'warnings':warnings,
  'interpretation':'ratio >1 means STM32 source time advances faster than host receive time in this recording',
  'limitations':['Not end-to-end physical sensor latency','Steady host clock is a reference, not a certified time standard','A changing serial backlog can mimic clock-rate error','No timestamp correction is applied; compare repeated recordings']}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('csv');p.add_argument('--output',required=True);p.add_argument('--warmup-s',type=float,default=5);p.add_argument('--bin-s',type=float,default=5);a=p.parse_args()
 if Path(a.csv).resolve()==Path(a.output).resolve():raise ValueError('Output must differ from input')
 with Path(a.csv).open() as f:rows=list(csv.DictReader(f))
 result=analyze(rows,a.warmup_s,a.bin_s)
 Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__':main()
