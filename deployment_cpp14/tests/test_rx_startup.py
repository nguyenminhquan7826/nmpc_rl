"""Reproduce stale USB/serial startup data without physical hardware."""
import binascii
import csv
import io
import os
from pathlib import Path
import pty
import subprocess
import sys
import tempfile
import termios
import time
import unittest

EXE=sys.argv.pop(1)

def frame(seq,ms,ticks=-7):
    payload=f'TEL,1,{seq},{ms},{seq*2},{ms-10},{ticks},0,0.0000,75,724,{seq*2},{ms-5},0,0,0,0,F1,3'
    return (payload+f',{binascii.crc_hqx(payload.encode(),0xffff):04X}\r\n').encode()

def wait_raw(slave,proc):
    deadline=time.monotonic()+2
    while termios.tcgetattr(slave)[3]&termios.ICANON:
        if proc.poll() is not None:raise AssertionError('Receiver exited before serial configuration')
        if time.monotonic()>deadline:raise AssertionError('Serial raw mode timeout')
        time.sleep(.005)

class StartupTests(unittest.TestCase):
    def test_old_frames_are_discarded_and_retained_log_replays(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);master,slave=pty.openpty()
            try:
                proc=subprocess.Popen([EXE,'--device',os.ttyname(slave),'--duration','1.2','--raw-log',str(tmp/'raw.log')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                wait_raw(slave,proc);ready=time.monotonic()
                os.write(master,frame(49290,985801))
                time.sleep(.15)
                # USB adapter delivers more old bytes after opening/configuration.
                os.write(master,b''.join(frame(49291+i,985821+20*i) for i in range(10)))
                while time.monotonic()-ready<.65:time.sleep(.005)
                os.write(master,b'old-fragment')
                time.sleep(.01)
                live=frame(87901,1758021)+frame(87902,1758041)
                os.write(master,b'-end\n'+live)
                out,err=proc.communicate(timeout=4)
                self.assertEqual(proc.returncode,0,err)
                rows=list(csv.DictReader(io.StringIO(out)))
                self.assertEqual([int(r['packet_seq']) for r in rows],[87901,87902])
                self.assertTrue(all(r['accepted']=='1' for r in rows))
                self.assertEqual((tmp/'raw.log').read_bytes(),live)
                self.assertIn('fault_latched=0',err)
                replay=subprocess.run([EXE,'--input',str(tmp/'raw.log')],capture_output=True,text=True,check=True)
                self.assertEqual(out,replay.stdout)
                os.set_blocking(master,False)
                try:sent=os.read(master,1024)
                except BlockingIOError:sent=b''
                self.assertEqual(sent,b'','Receiver must not transmit commands')
            finally:
                if proc.poll() is None:proc.kill();proc.wait()
                os.close(master);os.close(slave)

    def test_gap_after_capture_start_still_latches(self):
        master,slave=pty.openpty()
        try:
            proc=subprocess.Popen([EXE,'--device',os.ttyname(slave),'--duration','.4','--startup-discard-ms','0'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            wait_raw(slave,proc)
            os.write(master,frame(1,100)+frame(2,1000)+frame(3,1020))
            out,err=proc.communicate(timeout=2)
            self.assertEqual(proc.returncode,2,err)
            rows=list(csv.DictReader(io.StringIO(out)))
            self.assertEqual([r['status'] for r in rows],['initialized','integration_gap_restart_required','fault_latched_restart_required'])
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
            os.close(master);os.close(slave)

    def test_no_data_exits_within_bounded_window(self):
        master,slave=pty.openpty()
        try:
            started=time.monotonic()
            proc=subprocess.Popen([EXE,'--device',os.ttyname(slave),'--duration','.2'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            out,err=proc.communicate(timeout=3)
            self.assertEqual(proc.returncode,2,err)
            self.assertGreaterEqual(time.monotonic()-started,.5)
            self.assertEqual(len(list(csv.DictReader(io.StringIO(out)))),0)
            self.assertIn('CRC-valid=0',err)
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
            os.close(master);os.close(slave)

    def test_invalid_startup_duration_rejected(self):
        for value in ('-1','nan','inf','10001'):
            with self.subTest(value=value):
                result=subprocess.run([EXE,'--device','/dev/null','--startup-discard-ms',value],capture_output=True,text=True)
                self.assertEqual(result.returncode,1)

if __name__=='__main__':unittest.main()
