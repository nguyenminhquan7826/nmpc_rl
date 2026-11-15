"""PTY/replay contract for the actual C++ hardware frontend; no board required."""
import binascii,csv,io,os,pty,subprocess,sys,tempfile,time,unittest,termios,fcntl,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from test_online_ac import model_fixture,write_values
from test_fixed_nmpc import fixture
EXE=Path(os.environ.get('HARDWARE_RUNNER',str(ROOT/'build/hardware_runner')))
def setup(folder):
 _,_,m=model_fixture();m.update(horizon=[1],design=[0]);write_values(folder/'model','AC_MODEL_V1',m)
 c,_,_=fixture(N=5);c.update(Ts=[.08],max_speed=[.15],max_steering=[float(c['input_upper'][1])],servo_sign=[1],task_q=[20,20,100,20,600,600],task_r=[50,100],feature_length=[2],min_reference_speed=[.05],state_timeout_ms=[120],solve_budget_ms=[60],ack_timeout_ms=[100]);c['state_lower'][0]=0
 write_values(folder/'settings','NMPC_HARDWARE_V1',c)
 (folder/'route.csv').write_text('index,s_m,t_ref_s,vx_ref_mps,vy_ref_mps,psi_ref_rad,r_ref_radps,X_ref_m,Y_ref_m,kappa_1_per_m\n'+''.join(f'{i},{i*.01},{i*.1},0.1,0,0,0,{i*.01},0,0\n' for i in range(201)))
 return ['--model',str(folder/'model'),'--settings',str(folder/'settings'),'--reference',str(folder/'route.csv'),'--output',str(folder/'out')]
def wire(p):return (p+f',{binascii.crc_hqx(p.encode(),0xffff):04X}\r\n').encode()
def tel(seq,ms,error=0):
 return wire(f'TEL,2,{seq},{ms},{seq},{ms},0,0,0,75,0,{seq},{ms-2},0,0,0,0,34,3,0,0,0,0,{ms-7},{ms-2},10,5,{error}')
class Tests(unittest.TestCase):
 def test_native_session_contract(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);setup(p)
   r=subprocess.run([str(EXE.parent/'test_hardware_session'),str(p/'model'),str(p/'route.csv'),str(p/'settings')],capture_output=True,text=True)
   self.assertEqual(r.returncode,0,r.stdout+r.stderr)
 def test_replay_four_modes_never_learns(self):
  self.assertTrue(EXE.is_file(),'hardware_runner target missing')
  for mode in ['fixed','actor_adapt','critic_adapt','full_adapt']:
   with tempfile.TemporaryDirectory() as d:
    p=Path(d);args=setup(p);(p/'input.log').write_bytes(b''.join(tel(i+1,1000+40*i) for i in range(10)))
    r=subprocess.run([str(EXE),*args,'--input',str(p/'input.log'),'--mode',mode],capture_output=True,text=True)
    self.assertEqual(r.returncode,0,r.stderr);self.assertEqual((p/'out/tx.log').read_bytes(),b'')
    summary=(p/'out/summary.txt').read_text();self.assertIn('actor_updates=0',summary);self.assertIn('critic_updates=0',summary)
    self.assertTrue(list(csv.DictReader(io.StringIO((p/'out/control.csv').read_text()))));self.assertTrue((p/'out/final.model').is_file())
 def test_live_shadow_sends_no_bytes_and_stale_stops(self):
  self.assertTrue(EXE.is_file(),'hardware_runner target missing')
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);args=setup(p);master,slave=pty.openpty()
   try:
    proc=subprocess.Popen([str(EXE),*args,'--device',os.ttyname(slave),'--startup-discard-ms','0','--duration','1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    deadline=time.monotonic()+3
    while termios.tcgetattr(slave)[3]&termios.ICANON:
     self.assertLess(time.monotonic(),deadline);time.sleep(.01)
    os.write(master,tel(1,1000));time.sleep(.04);os.write(master,tel(2,1040))
    out,err=proc.communicate(timeout=4);self.assertEqual(proc.returncode,2,out+err)
    os.set_blocking(master,False)
    try:sent=os.read(master,4096)
    except BlockingIOError:sent=b''
    self.assertEqual(sent,b'');self.assertIn('telemetry_timeout',(p/'out/summary.txt').read_text())
   finally:
    if proc.poll() is None:proc.kill();proc.wait()
    os.close(master);os.close(slave)
 def test_drive_missing_ack_stops_and_never_sends_run(self):
  self.assertTrue(EXE.is_file(),'hardware_runner target missing')
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);args=setup(p);master,slave=pty.openpty()
   try:
    r=subprocess.run([str(EXE),*args,'--device',os.ttyname(slave),'--startup-discard-ms','0','--duration','.5','--drive','--x0','0','--y0','0','--yaw0-deg','0'],capture_output=True,text=True,timeout=4)
    self.assertEqual(r.returncode,2,r.stderr);sent=(p/'out/tx.log').read_text();self.assertIn(',STOP,',sent);self.assertNotIn(',RUN,',sent);self.assertIn('ack_timeout',(p/'out/summary.txt').read_text())
   finally:os.close(master);os.close(slave)
 def test_drive_acknowledged_transitions_and_fusion_fault_stop(self):
  for mode in ['fixed','actor_adapt','critic_adapt','full_adapt']:
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
    p=Path(d);args=setup(p)
    if mode=='fixed':
     (p/'route.csv').write_text('index,s_m,t_ref_s,vx_ref_mps,vy_ref_mps,psi_ref_rad,r_ref_radps,X_ref_m,Y_ref_m,kappa_1_per_m\n'+''.join(f'{i},{i*.01},{i*.1},0.1,0,{i*.01},0.1,{math.sin(i*.01)},{1-math.cos(i*.01)},1\n' for i in range(201)))
    master,slave=pty.openpty();os.set_blocking(master,False)
    proc=subprocess.Popen([str(EXE),*args,'--device',os.ttyname(slave),'--startup-discard-ms','0','--duration','1.3','--mode',mode,'--drive','--x0','0','--y0','0','--yaw0-deg','0'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
     start=time.monotonic();next_tel=start+.06;seq=1;buf=b'';last_seq=0;run_count=0;ready=False
     while proc.poll() is None and time.monotonic()-start<4:
      now=time.monotonic()
      if now>=next_tel and not(termios.tcgetattr(slave)[3]&termios.ICANON):
       ms=1000+int((now-start)*1000);os.write(master,tel(seq,ms,error=9 if run_count>=3 else 0));seq+=1;next_tel=now+.04
      try:buf+=os.read(master,4096)
      except BlockingIOError:pass
      while b'\n' in buf:
       line,buf=buf.split(b'\n',1);v=line.decode().strip().split(',')
       if v[0]!='CTL':continue
       self.assertEqual(int(v[-1],16),binascii.crc_hqx(','.join(v[:-1]).encode(),0xffff))
       req=int(v[2]);action=v[3];armed=int(action!='STOP');last_seq=req if action!='STOP' else last_seq
       if action=='RUN':run_count+=1
       ms=1000+int((time.monotonic()-start)*1000)
       os.write(master,wire(f'ACK,1,{req},{armed},0,{ms},{last_seq}'))
      time.sleep(.002)
     out,err=proc.communicate(timeout=2);self.assertEqual(proc.returncode,2,out+err)
     summary=(p/'out/summary.txt').read_text();self.assertIn('reason=fusion_unhealthy',summary);self.assertGreaterEqual(run_count,3)
     updates=int(dict(x.split('=',1) for x in summary.splitlines())['critic_updates'])
     self.assertEqual(updates>0,mode in ['critic_adapt','full_adapt'])
     self.assertEqual((p/'out/tx.log').read_text().strip().splitlines()[-1].split(',')[3],'STOP')
    finally:
     if proc.poll() is None:proc.kill();proc.wait()
     os.close(master);os.close(slave)
 def test_busy_serial_is_never_written_by_second_owner(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);args=setup(p);master,slave=pty.openpty()
   try:
    fcntl.flock(slave,fcntl.LOCK_EX|fcntl.LOCK_NB)
    r=subprocess.run([str(EXE),*args,'--device',os.ttyname(slave),'--drive','--x0','0','--y0','0','--yaw0-deg','0'],capture_output=True,text=True,timeout=3)
    self.assertEqual(r.returncode,1);self.assertIn('already owned',r.stderr);self.assertEqual((p/'out/tx.log').read_bytes(),b'')
   finally:os.close(master);os.close(slave)
 def test_solver_deadline_cancels_shadow_without_learning(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);args=setup(p)
   text=(p/'settings').read_text().replace('solve_budget_ms 1 60','solve_budget_ms 1 1e-12');(p/'settings').write_text(text)
   (p/'input.log').write_bytes(tel(1,1000))
   r=subprocess.run([str(EXE),*args,'--input',str(p/'input.log'),'--mode','full_adapt'],capture_output=True,text=True)
   self.assertEqual(r.returncode,2,r.stderr);self.assertIn('solve_deadline_or_stale_state',(p/'out/summary.txt').read_text());self.assertEqual((p/'out/tx.log').read_bytes(),b'')
if __name__=='__main__':unittest.main()
