"""PTY check for opt-in host read timestamps, unchanged raw bytes and replay."""
import binascii,csv,io,os,pty,subprocess,sys,tempfile,termios,time
from pathlib import Path
exe=sys.argv[1]
def frame(seq,ms):
 p=f'TEL,1,{seq},{ms},{seq},{ms},0,0,0,75,0,{seq},{ms},0,0,0,0,34,3'
 return (p+f',{binascii.crc_hqx(p.encode(),65535):04X}\r\n').encode()
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);master,slave=pty.openpty();proc=None
 try:
  proc=subprocess.Popen([exe,'--device',os.ttyname(slave),'--duration','.4','--startup-discard-ms','0','--host-timestamps','--raw-log',str(root/'capture.log')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
  deadline=time.monotonic()+2
  while termios.tcgetattr(slave)[3]&termios.ICANON:
   assert proc.poll() is None,'Receiver exited before raw configuration'
   assert time.monotonic()<deadline;time.sleep(.005)
  wire=frame(1,100)+frame(2,120)+frame(3,140)
  os.write(master,wire[:13]);time.sleep(.02);os.write(master,wire[13:-5]);time.sleep(.02);os.write(master,wire[-5:])
  out,err=proc.communicate(timeout=3);assert proc.returncode==0,err
  rows=list(csv.DictReader(io.StringIO(out)));assert len(rows)==3
  ns=[int(r['host_rx_monotonic_ns']) for r in rows];ids=[int(r['host_rx_batch_id']) for r in rows]
  assert ns==sorted(ns) and ns[0]>0 and ids==sorted(ids)
  assert float(rows[0]['host_rx_elapsed_s'])==0
  for r in rows:assert abs(float(r['host_rx_elapsed_s'])-(int(r['host_rx_monotonic_ns'])-ns[0])*1e-9)<1e-12
  for a,b in zip(rows,rows[1:]):
   if a['host_rx_batch_id']==b['host_rx_batch_id']:assert a['host_rx_monotonic_ns']==b['host_rx_monotonic_ns']
  assert (root/'capture.log').read_bytes()==wire
  replay=subprocess.run([exe,'--input',str(root/'capture.log')],capture_output=True,text=True,check=True)
  plain=list(csv.DictReader(io.StringIO(replay.stdout)))
  for a,b in zip(plain,rows):assert all(b[k]==v for k,v in a.items())
  assert 'host_rx_monotonic_ns' not in plain[0]
  bad=subprocess.run([exe,'--input',str(root/'capture.log'),'--host-timestamps'],capture_output=True,text=True)
  assert bad.returncode==1,'Must not fabricate historical host timestamps during replay'
  os.set_blocking(master,False)
  try:sent=os.read(master,1024)
  except BlockingIOError:sent=b''
  assert sent==b''
 finally:
  if proc is not None and proc.poll() is None:proc.kill();proc.wait()
  os.close(master);os.close(slave)
print('Host monotonic read timestamps, fragmentation, raw replay and RX-only PTY PASS')
