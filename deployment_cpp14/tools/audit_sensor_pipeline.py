"""Independent TEL log -> diagnostic CSV audit. Standard-library Python only.
Checks conversion/transport/accepted-pose arithmetic, not physical sensor truth
or the correctness of the receiver's accept/reject policy.
"""
import argparse,binascii,csv,json,math
from pathlib import Path

INTS={'tel_version':1,'packet_seq':2,'tx_ms':3,'enc_seq':4,'enc_ms':5,'enc_total':6,'enc_delta_ticks':7,'servo_command_deg':9,'pwm_command':10,'imu_seq':11,'imu_ms':12,'flags':18}
FLOATS={'speed_reported_mps':8,'linear_ax_mps2':13,'linear_ay_mps2':14,'gyro_z_raw_dps':15,'heading_raw_deg':16}
V2F={'gyro_x_raw_dps':19,'gyro_y_raw_dps':20,'roll_raw_deg':21,'pitch_raw_deg':22}
V2I={'imu_read_start_ms':23,'imu_read_end_ms':24,'enc_interval_ms':25,'imu_system_status':26,'imu_system_error':27}
def raw_frames(path):
 frames=[];invalid=0
 for line in Path(path).read_bytes().splitlines(keepends=True):
  try:
   if not line.endswith(b'\n') or len(line)>514:raise ValueError('framing')
   text=line.rstrip(b'\r\n');payload,crc=text.rsplit(b',',1)
   if len(crc)!=4 or binascii.crc_hqx(payload,65535)!=int(crc,16):raise ValueError('crc')
   f=text.decode('ascii').split(',')
   if f[0]!='TEL' or not ((f[1]=='1' and len(f)==20) or (f[1]=='2' and len(f)==29)):raise ValueError('version')
   for key,index in INTS.items():
    value=int(f[index])
    if key in ['tel_version','flags']:lo,hi=(1,2) if key=='tel_version' else (0,255)
    elif key=='enc_total' or key in ['servo_command_deg','pwm_command']:lo,hi=-2**31,2**31-1
    elif key=='enc_delta_ticks':lo,hi=-32768,32767
    else:lo,hi=0,2**32-1
    if not lo<=value<=hi:raise ValueError('integer range')
   for key,index in FLOATS.items():
    if not math.isfinite(float(f[index])):raise ValueError('nonfinite')
   int(f[17],16)
   if f[1]=='2':
    for key,index in V2F.items():
     if not math.isfinite(float(f[index])):raise ValueError('nonfinite')
    for key,index in V2I.items():
     if not 0<=int(f[index])<=(255 if key in ['imu_system_status','imu_system_error'] else 2**32-1):raise ValueError('integer range')
    if int(f[24])!=int(f[12]) or (int(f[24])-int(f[23]))%2**32>=2**31:raise ValueError('read timing')
   frames.append(f)
  except (ValueError,UnicodeError,IndexError):invalid+=1
 return frames,invalid
def audit(frames,rows,gyro_sign,heading_sign=-1,x0=0.,y0=0.,yaw0=0.):
 errors=[];x=x0;y=y0;yaw=yaw0;travel=0.;vx=0.;rate=0.;last=None
 if len(frames)!=len(rows):errors.append(f'Frame/CSV row count differs: {len(frames)}/{len(rows)}')
 required=set(INTS)|set(FLOATS)|set(V2F)|set(V2I)|{'calib_hex','accepted','X_est_m','Y_est_m','yaw_est_rad','r_measured_radps','vx_measured_mps','signed_travel_m'}
 if rows and not required.issubset(rows[0]):raise ValueError('CSV lacks diagnostic columns; capture with --diagnostics')
 def check_float(row,col,expected,n):
  try:ok=math.isclose(float(row[col]),expected,rel_tol=1e-11,abs_tol=1e-10)
  except (ValueError,KeyError):ok=False
  if not ok:errors.append(f'row {n}: {col} mismatch')
 for n,(f,row) in enumerate(zip(frames,rows),1):
  for col,i in INTS.items():
   if row[col]!=str(int(f[i])):errors.append(f'row {n}: {col} mismatch')
  if int(row['calib_hex'],16)!=int(f[17],16):errors.append(f'row {n}: calibration mismatch')
  for col,i in FLOATS.items():check_float(row,col,float(f[i]),n)
  if f[1]=='2':
   for col,i in V2F.items():check_float(row,col,float(f[i]),n)
   for col,i in V2I.items():
    if row[col]!=str(int(f[i])):errors.append(f'row {n}: {col} mismatch')
  else:
   if any(row[c]!='' for c in list(V2F)+list(V2I)):errors.append(f'row {n}: v1 diagnostics must be blank')
  if row['accepted']=='1':
   if last is not None:
    diff=math.radians(float(f[16])-float(last[16]));turn=heading_sign*math.atan2(math.sin(diff),math.cos(diff))
    ticks=(int(f[6])-int(last[6]))%(2**32);ticks=ticks if ticks<2**31 else ticks-2**32
    ds=ticks*(2*math.pi*.0325/(11*4*19*(38/13)))
    x+=ds*math.cos(yaw+turn/2);y+=ds*math.sin(yaw+turn/2);travel+=ds;yaw+=turn
   vx=float(f[8]);rate=gyro_sign*math.radians(float(f[15]));last=f
  elif row['accepted']!='0':errors.append(f'row {n}: accepted must be 0/1')
  for col,v in [('X_est_m',x),('Y_est_m',y),('yaw_est_rad',yaw),('r_measured_radps',rate),('vx_measured_mps',vx),('signed_travel_m',travel)]:check_float(row,col,v,n)
 return {'mapping_and_pose_passed':not errors,'frames':len(frames),'csv_rows':len(rows),'mismatch_count':len(errors),'mismatch_examples':errors[:20]}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('log');p.add_argument('csv');p.add_argument('--gyro-sign',type=int,choices=(-1,1),required=True);p.add_argument('--heading-sign',type=int,choices=(-1,1),default=-1);p.add_argument('--x0',type=float,default=0);p.add_argument('--y0',type=float,default=0);p.add_argument('--yaw0-deg',type=float,default=0);p.add_argument('--output',required=True);a=p.parse_args()
 if Path(a.output).resolve() in [Path(a.log).resolve(),Path(a.csv).resolve()]:raise ValueError('Output must differ from inputs')
 f,bad=raw_frames(a.log)
 with Path(a.csv).open() as handle:rows=list(csv.DictReader(handle))
 result=audit(f,rows,a.gyro_sign,a.heading_sign,a.x0,a.y0,math.radians(a.yaw0_deg))
 result['invalid_or_incomplete_raw_lines']=bad;result['passed']=result['mapping_and_pose_passed'] and bad==0 and len(f)>0
 v2=[r for r in f if r[1]=='2']
 if v2:result['v2_quality']={'max_abs_sensor_roll_deg':max(abs(float(r[21])) for r in v2),'max_abs_sensor_pitch_deg':max(abs(float(r[22])) for r in v2),'system_error_nonzero_frames':sum(int(r[27])!=0 for r in v2),'read_duration_ms_range':[min((int(r[24])-int(r[23]))%2**32 for r in v2),max((int(r[24])-int(r[23]))%2**32 for r in v2)],'encoder_interval_ms_range':[min(int(r[25]) for r in v2),max(int(r[25]) for r in v2)]}
 result['limitations']=['Raw fields are sensor-reported SI/degree values, not original register counts','Does not validate physical angle, mounting, calibration or source-clock accuracy','Accept/reject decisions are consumed from CSV, not independently reimplemented','Fixed production wheel geometry is assumed for pose arithmetic']
 Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2,allow_nan=False))
 return 0 if result['passed'] else 2
if __name__=='__main__':raise SystemExit(main())
