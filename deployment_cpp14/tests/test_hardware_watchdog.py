"""PTY contract: neutral commands only, no motor or board required."""
import binascii, os, pty, subprocess, sys, termios, time, unittest

EXE = sys.argv.pop(1)
def wire(payload):
    return (payload + f',{binascii.crc_hqx(payload.encode(), 0xffff):04X}\r\n').encode()
def tel(seq, ms):
    return wire(f'TEL,2,{seq},{ms},{seq},{ms},0,0,0,75,0,{seq},{ms-2},0,0,0,0,34,3,0,0,0,0,{ms-7},{ms-2},10,5,0')

class Tests(unittest.TestCase):
    def test_requires_explicit_confirmation(self):
        p = subprocess.run([EXE], capture_output=True, text=True, timeout=2)
        self.assertEqual(p.returncode, 1)
        self.assertIn('--confirm-disconnected', p.stderr)

    def test_neutral_watchdog_and_failure(self):
        for expires in (True, False):
            with self.subTest(expires=expires):
                master, slave = pty.openpty()
                os.set_blocking(master, False)
                p = subprocess.Popen([EXE, '--device', os.ttyname(slave),
                                      '--confirm-disconnected'],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    start = time.monotonic()
                    next_tel = start
                    seq = 1
                    last = 0
                    armed = False
                    last_run = None
                    actions = []
                    runs = []
                    buf = b''
                    while p.poll() is None and time.monotonic() - start < 4:
                        now = time.monotonic()
                        ms = 1000 + int((now-start)*1000)
                        if expires and armed and last_run is not None and now-last_run >= .3:
                            armed = False
                        if actions and now >= next_tel:
                            os.write(master, tel(seq, ms))
                            seq += 1
                            next_tel = now+.04
                        try:
                            buf += os.read(master, 4096)
                        except BlockingIOError:
                            pass
                        while b'\n' in buf:
                            line, buf = buf.split(b'\n', 1)
                            v = line.decode().strip().split(',')
                            if v[0] != 'CTL' or termios.tcgetattr(slave)[3]&termios.ICANON:
                                continue
                            self.assertEqual(int(v[-1], 16), binascii.crc_hqx(','.join(v[:-1]).encode(), 0xffff))
                            req, action = int(v[2]), v[3]
                            self.assertEqual(float(v[4]), 0)
                            self.assertEqual(int(v[5]), 75)
                            actions.append(action)
                            reason = 0
                            if action == 'STOP':
                                armed = False
                            elif action == 'ARM':
                                armed = True
                                last = req
                            elif action == 'RUN':
                                runs.append(now)
                                if armed:
                                    last = req
                                    last_run = now
                                else:
                                    reason = 3
                            os.write(master, wire(f'ACK,1,{req},{int(armed)},{reason},{ms},{last}'))
                        time.sleep(.001)
                    out, err = p.communicate(timeout=2)
                    self.assertEqual(p.returncode, 0 if expires else 2, out+err)
                    self.assertEqual(actions, ['STOP','ARM','RUN','RUN','STOP'])
                    self.assertGreaterEqual(runs[1]-runs[0], .45)
                    self.assertIn('final_stop_confirmed=1', out)
                    self.assertIn('result=PASS_NEUTRAL_DISARM' if expires else 'result=FAIL', out)
                finally:
                    if p.poll() is None:
                        p.kill()
                        p.wait()
                    os.close(master)
                    os.close(slave)

if __name__ == '__main__':
    unittest.main()
