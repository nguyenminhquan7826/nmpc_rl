// Neutral-command-only integration test. No CasADi, model, or learning.
#include "hardware_control.hpp"
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

namespace {
using namespace deployment;
using Clock = std::chrono::steady_clock;
using Ms = std::chrono::milliseconds;
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }
void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
class Bench {
  int fd = -1;
  bool owned = false, confirmed = false, discard = false;
  std::uint32_t seq = UINT32_MAX;
  std::string line;
  Clock::time_point last_tel{};
  Telemetry latest{};
  std::size_t healthy = 0;
  Odometry odom;
  bool monitor = false;

  void send(const std::string &frame) {
    require(owned, "serial not owned");
    std::size_t pos = 0;
    const auto deadline = Clock::now() + Ms(20);
    while (pos < frame.size()) {
      require(Clock::now() < deadline, "serial write deadline");
      const auto n = ::write(fd, frame.data() + pos, frame.size() - pos);
      if (n > 0) pos += static_cast<std::size_t>(n);
      else {
        require(n < 0 && (errno == EAGAIN || errno == EINTR), "serial write failed");
        pollfd p{fd, POLLOUT, 0};
        ::poll(&p, 1, 1);
      }
    }
  }

  // Read whole chunks, retain partial lines, and never run any controller.
  // On shutdown ignore telemetry health so STOP ACK can still be accepted.
  bool pump(Clock::time_point deadline, std::uint32_t expected, Ack *out,
            bool shutdown = false) {
    if (!shutdown) require(!interrupted, "operator interrupted");
    if (Clock::now() >= deadline) return false;
    pollfd p{fd, POLLIN, 0};
    int ready = ::poll(&p, 1, 2);
    if (ready < 0 && errno == EINTR) return false;
    require(ready >= 0 && !(p.revents & (POLLERR | POLLHUP | POLLNVAL)),
            "serial disconnected");
    bool matched = false;
    if (p.revents & POLLIN) {
      char bytes[4096];
      auto n = ::read(fd, bytes, sizeof(bytes));
      if (n < 0 && (errno == EAGAIN || errno == EINTR)) return false;
      require(n > 0, "serial read failed");
      for (ssize_t i = 0; i < n; ++i) {
        char c = bytes[i];
        if (c == '\n') {
          if (!discard) {
            if (line.compare(0, 4, "ACK,") == 0) {
              Ack a{};
              if (parse_ack(line, a)) {
                std::cout << line << '\n';
                if (out && a.seq == expected && Clock::now() <= deadline && !matched) {
                  *out = a;
                  matched = true;
                }
              }
            } else if (line.compare(0, 9, "UARTDIAG,") == 0) {
              std::cout << line << '\n';
            } else if (!shutdown && monitor && line.compare(0, 4, "TEL,") == 0) {
              Telemetry t{};
              std::string status;
              require(parse(line, t, status) && t.protocol_version == 2,
                      "invalid TEL2");
              require(t.system_status == 5 && t.system_error == 0 &&
                      odom.update(t, status), "unhealthy telemetry");
              require(std::abs(t.speed) <= .02 &&
                      std::abs(t.gyro_dps * pi / 180) <= .1,
                      "vehicle not stationary");
              require(t.pwm == 0 && t.servo_deg == 75,
                      "non-neutral reported output");
              if (!healthy || t.imu_ms != latest.imu_ms) {
                last_tel = Clock::now();
                ++healthy;
              }
              latest = t;
            }
          }
          line.clear();
          discard = false;
        } else if (c != '\r' && !discard) {
          if (line.size() < 512) line += c;
          else { line.clear(); discard = true; }
        }
      }
    }
    if (!shutdown && monitor && healthy)
      require(Clock::now() - last_tel <= Ms(120), "telemetry timeout");
    return matched;
  }
  Ack command(const std::string &action, bool shutdown = false) {
    if (!shutdown) require(!interrupted, "operator interrupted");
    const auto id = ++seq;
    const auto frame = command_frame(id, action, 0, 75);
    confirmed = false;
    send((action == "STOP" ? "\r\n" : "") + frame);
    std::cout << "tx_" << action << '=' << id << '\n';
    const auto deadline = Clock::now() + Ms(100);
    Ack a{};
    while (Clock::now() < deadline) {
      if (pump(deadline, id, &a, shutdown)) return a;
    }
    throw std::runtime_error("ACK timeout: " + action);
  }
  void observe(Ms duration) {
    const auto deadline = Clock::now() + duration;
    const auto before = healthy;
    while (Clock::now() < deadline) pump(deadline, 0, nullptr);
    require(healthy > before, "no fresh telemetry during observation");
  }
public:
  void open(const std::string &device) {
    fd = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    require(fd >= 0, "cannot open serial");
    require(::flock(fd, LOCK_EX | LOCK_NB) == 0, "serial already owned");
    termios t{};
    require(tcgetattr(fd, &t) == 0, "tcgetattr failed");
    cfmakeraw(&t);
    require(cfsetispeed(&t, B115200) == 0 && cfsetospeed(&t, B115200) == 0,
            "baud setup failed");
    t.c_cflag |= CLOCAL | CREAD;
    t.c_cflag &= ~(CSTOPB | CRTSCTS);
    require(tcsetattr(fd, TCSANOW, &t) == 0, "tcsetattr failed");
    require(::ioctl(fd, TIOCEXCL) == 0, "cannot reserve serial");
    owned = true;
    require(tcflush(fd, TCIFLUSH) == 0, "input flush failed");
  }
  void stop() {
    if (!owned || confirmed) return;
    const Ack a = command("STOP", true);
    require(a.reason == 0 && !a.armed, "STOP not confirmed");
    confirmed = true;
    std::cout << "final_stop_confirmed=1\n";
  }
  void run() {
    const Ack sync = command("STOP");
    require(sync.reason == 0 && !sync.armed, "initial STOP not confirmed");
    seq = sync.last_seq;
    monitor = true;
    observe(Ms(200));
    require(healthy >= 3, "insufficient stationary telemetry");
    const Ack arm = command("ARM");
    require(arm.armed && arm.reason == 0 && arm.last_seq == arm.seq,
            "ARM not accepted");
    const Ack run = command("RUN");
    require(run.armed && run.reason == 0 && run.last_seq == run.seq,
            "neutral RUN not accepted");
    std::cout << "neutral_run_accepted=1\n";
    const auto silent_start = Clock::now();
    // Intentionally no transmit here; telemetry continues to be monitored.
    observe(Ms(450));
    const double silence_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - silent_start).count();
    const Ack probe = command("RUN");  // neutral only, must be rejected
    const auto mcu_ms = static_cast<std::uint32_t>(probe.ms - run.ms);
    std::cout << "silence_host_ms=" << silence_ms << '\n'
              << "silence_mcu_ms=" << mcu_ms << '\n'
              << "probe_armed=" << probe.armed << '\n'
              << "probe_reason=" << probe.reason << '\n'
              << "healthy_samples=" << healthy << '\n';
    require(mcu_ms >= 300 && mcu_ms <= 1000, "unexpected MCU elapsed time");
    require(!probe.armed && probe.reason == 3 && probe.last_seq == run.seq,
            "watchdog probe did not confirm disarmed");
  }
  ~Bench() {
    if (owned && !confirmed) {
      try { send("\r\n" + command_frame(++seq, "STOP", 0, 75)); }
      catch (...) {}
    }
    if (owned) ::ioctl(fd, TIOCNXCL);
    if (fd >= 0) ::close(fd);
  }
};
}

int main(int argc, char **argv) {
  std::string device = "/dev/ttyACM0";
  bool disconnected = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--device" && i + 1 < argc) device = argv[++i];
    else if (arg == "--confirm-disconnected") disconnected = true;
    else if (arg == "--help") {
      std::cout << "hardware_watchdog_check --confirm-disconnected [--device /dev/ttyACM0]\n"
                   "Motor AND servo power must be disconnected; logic stays powered.\n"
                   "Only neutral ARM/RUN/STOP are supported.\n";
      return 0;
    } else {
      std::cerr << "Unknown/incomplete argument: " << arg << '\n';
      return 1;
    }
  }
  if (!disconnected) {
    std::cerr << "Disconnect motor and servo power; explicit --confirm-disconnected required.\n";
    return 1;
  }
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  Bench bench;
  try {
    bench.open(device);
    bench.run();
    bench.stop();
    std::cout << "result=PASS_NEUTRAL_DISARM\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "failure=" << e.what() << '\n';
    try { bench.stop(); }
    catch (const std::exception &stop_error) {
      std::cerr << "final_stop_confirmed=0\nstop_error=" << stop_error.what() << '\n';
    }
    std::cout << "result=FAIL\n";
    return 2;
  }
}
