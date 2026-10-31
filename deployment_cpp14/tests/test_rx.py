"""Host integration test; no physical serial port and no motor commands."""
import binascii,csv,io,os,pty,subprocess,sys,tempfile,time
from pathlib import Path
exe=sys.argv[1]
def frame(seq,ms,count):
    p=f'TEL,1,{seq},{ms},{seq},{ms},{count},12,0.1000,75,100,{seq},{ms},0,0,0,0,FF,3'
    return (p+f',{binascii.crc_hqx(p.encode(),0xffff):04X}\r\n').encode()
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp);log=tmp/'tel.log'
    valid=frame(1,100,1000)+frame(2,120,1024)+frame(4,160,1072)
    log.write_bytes(b'x'*600+b'\n'+valid+frame(4,160,1072)+b'TEL,bad\n')
    r=subprocess.run([exe,'--input',str(log)],capture_output=True,text=True,check=True)
    rows=list(csv.DictReader(io.StringIO(r.stdout)))
    assert len(rows)==4 and rows[-1]['status']=='duplicate_or_reordered_packet'
    assert abs(float(rows[2]['signed_travel_m'])-72*2*3.141592653589793*.0325/(836*38/13))<1e-12
    assert 'invalid_frames=2' in r.stderr
    master,slave=pty.openpty();path=os.ttyname(slave)
    proc=subprocess.Popen([exe,'--device',path,'--duration','0.8','--raw-log',str(tmp/'capture.log')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    # Wait for termios raw mode so line discipline cannot echo test bytes.
    import termios
    deadline=time.monotonic()+2
    while termios.tcgetattr(slave)[3]&termios.ICANON:
        assert time.monotonic()<deadline
        time.sleep(.01)
    os.write(master,valid[:17]);os.write(master,valid[17:])
    out,err=proc.communicate(timeout=3)
    assert proc.returncode==0,(out,err)
    assert len(list(csv.DictReader(io.StringIO(out))))==3
    assert (tmp/'capture.log').read_bytes()==valid
    # RX-only executable must emit no application data toward STM32.
    os.set_blocking(master,False)
    try: sent=os.read(master,1024)
    except BlockingIOError: sent=b''
    assert sent==b'',sent
    os.close(master);os.close(slave)
print('Replay/framing, dropped packets, raw capture and fragmented RX-only PTY tests PASS')
