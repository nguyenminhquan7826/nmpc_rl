import binascii,csv,io,subprocess,sys,tempfile,math
from pathlib import Path
exe=sys.argv[1]
def frame(payload):return (payload+f',{binascii.crc_hqx(payload.encode(),65535):04X}\r\n').encode()
v1='TEL,1,1,100,1,99,-7,-1,-0.0125,75,0,1,96,0,0,10.0625,359.94,34,3'
v2='TEL,2,2,140,2,139,-7,0,0.0000,75,0,2,136,0,0,10.0625,359.9375,34,3,-0.1250,0.2500,2.0625,-3.1250,131,136,10,5,0'
with tempfile.TemporaryDirectory() as directory:
 p=Path(directory)/'v2.log';p.write_bytes(frame(v1)+frame(v2))
 r=subprocess.run([exe,'--input',str(p),'--diagnostics','--gyro-sign','1'],capture_output=True,text=True)
 assert r.returncode==0,r.stderr
 rows=list(csv.DictReader(io.StringIO(r.stdout)));assert len(rows)==2
 assert rows[0]['tel_version']=='1' and rows[0]['gyro_x_raw_dps']==''
 assert rows[1]['tel_version']=='2'
 for c,v in {'gyro_x_raw_dps':-.125,'gyro_y_raw_dps':.25,'roll_raw_deg':2.0625,'pitch_raw_deg':-3.125,'heading_raw_deg':359.9375,'imu_read_start_ms':131,'imu_read_end_ms':136,'enc_interval_ms':10,'imu_system_status':5,'imu_system_error':0}.items():assert abs(float(rows[1][c])-v)<1e-12,(c,rows[1][c])
 assert abs(float(rows[1]['r_measured_radps'])-10.0625*math.pi/180)<1e-12
 import importlib.util,copy
 spec=importlib.util.spec_from_file_location('pipeline',Path(__file__).parents[1]/'tools/audit_sensor_pipeline.py')
 audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
 raw,bad_count=audit.raw_frames(p);assert bad_count==0
 assert audit.audit(raw,rows,1)['mapping_and_pose_passed']
 changed=copy.deepcopy(rows);changed[1]['gyro_x_raw_dps']='0.5'
 assert not audit.audit(raw,changed,1)['mapping_and_pose_passed']
 changed=copy.deepcopy(rows);changed[1]['yaw_est_rad']='1.0'
 assert not audit.audit(raw,changed,1)['mapping_and_pose_passed']
 for bad in [v2.replace('TEL,2','TEL,3'),v2.replace(',131,136,10,5,0',',131,135,10,5,0'),v2.replace('TEL,2','TEL,1')]:
  p.write_bytes(frame(bad));z=subprocess.run([exe,'--input',str(p),'--diagnostics'],capture_output=True,text=True);assert z.returncode==2 and 'invalid_frames=1' in z.stderr
print('TEL v1/v2 coexistence, diagnostic precision and invalid version/timing PASS')
