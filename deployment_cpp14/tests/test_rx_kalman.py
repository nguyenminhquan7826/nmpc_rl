"""Optional KF must preserve measured odometry and skip rejected/duplicate IMU."""
import binascii,csv,io,subprocess,sys,tempfile,math
from pathlib import Path
exe=sys.argv[1]
def frame(seq,ms,imu_ms,yaw,gyro):
 p=f'TEL,1,{seq},{ms},{seq},{ms},0,0,0,75,0,{imu_ms},{imu_ms},0,0,{gyro},{yaw},FF,3'
 return (p+f',{binascii.crc_hqx(p.encode(),65535):04X}\r\n').encode()
with tempfile.TemporaryDirectory() as directory:
 p=Path(directory)/'data.log'
 p.write_bytes(frame(1,100,100,.1,10)+frame(2,120,120,359.9,10)+frame(3,140,120,359.9,10)+frame(3,140,120,359.9,10)+frame(4,160,160,359.5,10))
 def replay(args):
  r=subprocess.run([exe,'--input',str(p),'--gyro-sign','1',*args],capture_output=True,text=True)
  assert r.returncode==0,r.stderr
  return list(csv.DictReader(io.StringIO(r.stdout)))
 plain=replay([]);filtered=replay(['--yaw-kalman'])
 assert [r['kf_status'] for r in filtered]==['initialized','updated','duplicate_imu','odometry_rejected','updated']
 for a,b in zip(plain,filtered):
  assert all(b[k]==v for k,v in a.items()),'Must retain measured columns'
 assert filtered[2]['yaw_kf_rad']=='' and filtered[3]['yaw_kf_rad']==''
 assert abs(float(filtered[1]['yaw_kf_rad'])-float(filtered[1]['yaw_est_rad']))<1e-10
 for option in ['--kf-heading-sigma-deg','--kf-gyro-sigma-dps','--kf-bias-rw-dps']:
  for value in ['0','-1','nan','inf']:
   r=subprocess.run([exe,'--input',str(p),'--yaw-kalman',option,value],capture_output=True,text=True)
   assert r.returncode==1,(option,value,r.stderr)
 p.write_bytes(frame(1,100,100,0,0)+frame(2,140,100,0,0)+frame(3,320,320,0,0)+frame(4,340,340,0,0))
 r=subprocess.run([exe,'--input',str(p),'--yaw-kalman'],capture_output=True,text=True)
 rows=list(csv.DictReader(io.StringIO(r.stdout)))
 assert r.returncode==2,r.stderr
 assert [x['kf_status'] for x in rows]==['initialized','duplicate_imu','filter_gap_restart_required','filter_gap_restart_required']
 assert all(x['accepted']=='1' for x in rows)
print('Optional KF CLI, wrap, duplicate/rejected samples and invalid settings PASS')
