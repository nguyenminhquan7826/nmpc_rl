"""Heading is clockwise; gyro Z is counterclockwise, as in rotation_03."""
import binascii
import csv
import io
import math
from pathlib import Path
import subprocess
import sys
import tempfile

exe = sys.argv[1]
def frame(seq, ms, yaw, gyro):
    payload = f'TEL,1,{seq},{ms},{seq},{ms},0,0,0,75,0,{seq},{ms},0,0,{gyro},{yaw},FF,3'
    return (payload + f',{binascii.crc_hqx(payload.encode(), 65535):04X}\r\n').encode()
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory)/'rotation.log'
    # 0.2 degrees CCW in 20 ms = +10 deg/s; heading crosses 0.
    path.write_bytes(frame(1,100,0.1,10)+frame(2,120,359.9,10))
    def replay(*args):
        p = subprocess.run([exe,'--input',str(path),*args],capture_output=True,text=True)
        assert p.returncode == 0, p.stderr
        return list(csv.DictReader(io.StringIO(p.stdout)))
    corrected = replay('--gyro-sign','1')
    assert abs(float(corrected[0]['r_measured_radps'])-10*math.pi/180)<1e-12
    yaw = float(corrected[-1]['yaw_est_rad'])
    rate = float(corrected[-1]['r_measured_radps'])
    assert abs(yaw/.02-rate)<1e-12, 'Yaw derivative and gyro must agree'
    legacy = replay()
    assert float(legacy[-1]['r_measured_radps']) < 0, 'Preserve legacy default'
    assert abs(float(legacy[-1]['yaw_est_rad'])-yaw)<1e-12
    for value in ('-1','1'):
        for options in (['--imu-ccw','--gyro-sign',value],['--gyro-sign',value,'--imu-ccw']):
            rows = replay(*options)
            assert abs(float(rows[-1]['r_measured_radps'])-int(value)*10*math.pi/180)<1e-12
    for value in ('0','2','nan','inf','1.1'):
        p = subprocess.run([exe,'--input',str(path),'--gyro-sign',value],capture_output=True,text=True)
        assert p.returncode == 1, (value,p.stderr)
print('Independent gyro sign, wrap, legacy convention and invalid configuration PASS')
