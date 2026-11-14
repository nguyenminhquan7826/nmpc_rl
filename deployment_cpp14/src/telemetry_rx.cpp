#include "telemetry.hpp"
#include "yaw_kalman.hpp"
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <poll.h>
#include <termios.h>
#include <unistd.h>
namespace {
volatile std::sig_atomic_t stopped = 0;
void stop(int) { stopped = 1; }
struct Port {
  int fd = -1;
  ~Port() {
    if (fd >= 0)
      ::close(fd);
  }
};
double number(const std::string &s) {
  std::size_t i = 0;
  double v = std::stod(s, &i);
  if (i != s.size())
    throw std::runtime_error("invalid number");
  return v;
}
} // namespace
int main(int argc, char **argv) {
  try {
    std::string input, device, output, raw_log;
    double duration = 30, startup_discard_ms = 500;
    deployment::Config config;
    bool kalman_enabled = false, host_timestamps = false, diagnostics = false;
    deployment::YawKalmanConfig kalman_config;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--diagnostics") {
        diagnostics = true; continue;
      }
      if (arg == "--host-timestamps") {
        host_timestamps = true;
        continue;
      }
      if (arg == "--yaw-kalman") {
        kalman_enabled = true;
        continue;
      }
      if (arg == "--imu-ccw") {
        config.imu_clockwise = false;
        continue;
      }
      if (arg == "--help") {
        std::cout
            << "RX only: --input TEL.log | --device /dev/ttyUSB0 [--output "
               "pose.csv] "
               "[--raw-log TEL.log] [--duration 30] [--startup-discard-ms 500] "
               "[--x0 0 --y0 "
               "0 --yaw0-deg 0] [--imu-ccw] [--gyro-sign 1|-1] [--yaw-kalman] "
               "[--host-timestamps] [--diagnostics]\n"
               "[--kf-heading-sigma-deg 1] [--kf-gyro-sigma-dps 0.35] "
               "[--kf-bias-rw-dps 0.01]\n";
        return 0;
      }
      if (i + 1 >= argc)
        throw std::runtime_error("missing option value");
      std::string value = argv[++i];
      if (arg == "--input")
        input = value;
      else if (arg == "--device")
        device = value;
      else if (arg == "--output")
        output = value;
      else if (arg == "--raw-log")
        raw_log = value;
      else if (arg == "--duration")
        duration = number(value);
      else if (arg == "--startup-discard-ms")
        startup_discard_ms = number(value);
      else if (arg == "--gyro-sign") {
        if (value != "1" && value != "-1")
          throw std::runtime_error("gyro-sign must be 1 or -1");
        config.gyro_sign = value == "1" ? 1 : -1;
      } else if (arg == "--kf-heading-sigma-deg")
        kalman_config.heading_sigma_deg = number(value);
      else if (arg == "--kf-gyro-sigma-dps")
        kalman_config.gyro_sigma_dps = number(value);
      else if (arg == "--kf-bias-rw-dps")
        kalman_config.bias_rw_dps = number(value);
      else if (arg == "--x0")
        config.x0 = number(value);
      else if (arg == "--y0")
        config.y0 = number(value);
      else if (arg == "--yaw0-deg")
        config.yaw0 = number(value) * deployment::pi / 180.;
      else
        throw std::runtime_error("unknown option: " + arg);
    }
    if (input.empty() == device.empty())
      throw std::runtime_error("select exactly one --input or --device");
    if (host_timestamps && !input.empty())
      throw std::runtime_error(
          "--host-timestamps requires --device; historical receive times "
          "cannot be reconstructed from --input");
    if (!(duration > 0 && duration < 86400))
      throw std::runtime_error("duration must be in (0,86400) seconds");
    if (!(startup_discard_ms >= 0 && startup_discard_ms <= 10000))
      throw std::runtime_error(
          "startup-discard-ms must be finite in [0,10000]");
    if ((!output.empty() && (output == input || output == raw_log)) ||
        (!raw_log.empty() && raw_log == input))
      throw std::runtime_error("input/output paths must differ");
    for (const auto &path : {output, raw_log})
      if (!path.empty() && std::ifstream(path).good())
        throw std::runtime_error("refusing to overwrite: " + path);
    deployment::YawKalman kalman(kalman_config);
    if (kalman_enabled)
      std::cerr << "Experimental yaw KF: heading_sigma_deg="
                << kalman_config.heading_sigma_deg
                << " gyro_sigma_dps=" << kalman_config.gyro_sigma_dps
                << " bias_rw_dps=" << kalman_config.bias_rw_dps
                << " heading_clockwise=" << config.imu_clockwise
                << " gyro_sign_override=" << config.gyro_sign
                << " (0=legacy)\n";
    bool kf_initialized = false, kf_fault = false;
    std::uint32_t kf_ms = 0, kf_seq = 0;
    std::ofstream csv, raw;
    if (!output.empty()) {
      csv.open(output);
      if (!csv)
        throw std::runtime_error("cannot open output");
    }
    if (!raw_log.empty()) {
      raw.open(raw_log, std::ios::binary);
      if (!raw)
        throw std::runtime_error("cannot open raw log");
    }
    auto &out = output.empty() ? std::cout : csv;
    out << "packet_seq,tx_ms,enc_ms,imu_ms,accepted,status,X_est_m,Y_est_m,yaw_"
           "est_rad,vx_measured_"
           "mps,r_measured_radps,signed_travel_m,enc_total,flags,calib_hex"
        << (kalman_enabled ? ",kf_status,yaw_kf_rad,gyro_bias_kf_radps,r_bias_"
                             "corrected_radps"
                           : "")
        << (host_timestamps
                ? ",host_rx_monotonic_ns,host_rx_elapsed_s,host_rx_batch_id"
                : "")
        << (diagnostics ? ",tel_version,enc_seq,imu_seq,enc_delta_ticks,speed_reported_mps,servo_command_deg,pwm_command,linear_ax_mps2,linear_ay_mps2,gyro_z_raw_dps,heading_raw_deg,gyro_x_raw_dps,gyro_y_raw_dps,roll_raw_deg,pitch_raw_deg,imu_read_start_ms,imu_read_end_ms,enc_interval_ms,imu_system_status,imu_system_error" : "")
        << '\n'
        << std::setprecision(15);
    deployment::Odometry odom(config);
    std::string buffer;
    bool discard = false;
    unsigned good = 0, bad = 0, rejected = 0;
    std::uint64_t host_read_ns = 0, host_origin_ns = 0, host_batch_id = 0;
    bool host_origin_set = false;
    auto consume = [&](const char *data, std::size_t size) {
      if (raw)
        raw.write(data, static_cast<std::streamsize>(size));
      for (std::size_t i = 0; i < size; ++i) {
        char ch = data[i];
        if (ch != '\n') {
          if (!discard) {
            buffer += ch;
            if (buffer.size() > 512) {
              discard = true;
              buffer.clear();
            }
          }
          continue;
        }
        if (discard) {
          ++bad;
          discard = false;
          buffer.clear();
          continue;
        }
        deployment::Telemetry p{};
        std::string status;
        if (!deployment::parse(buffer, p, status)) {
          ++bad;
          std::cerr << "Rejected frame: " << status << "\n";
          buffer.clear();
          continue;
        }
        ++good;
        if (host_timestamps && !host_origin_set) {
          host_origin_ns = host_read_ns;
          host_origin_set = true;
        }
        bool accepted = odom.update(p, status);
        if (!accepted)
          ++rejected;
        const auto &pose = odom.pose();
        out << p.sequence << ',' << p.tx_ms << ',' << p.enc_ms << ','
            << p.imu_ms << ',' << accepted << ',' << status << ',' << pose.x
            << ',' << pose.y << ',' << pose.yaw << ',' << pose.vx << ','
            << pose.r << ',' << pose.travel << ',' << p.total_ticks << ','
            << p.flags << ',' << std::hex << p.calibration << std::dec;
        if (kalman_enabled) {
          std::string state = "odometry_rejected";
          bool fresh = false;
          if (accepted) {
            if (kf_fault)
              state = "filter_gap_restart_required";
            else if (!kf_initialized) {
              kalman.initialize(pose.yaw);
              kf_initialized = true;
              fresh = true;
              state = "initialized";
            } else if (p.imu_ms == kf_ms || p.imu_sequence == kf_seq)
              state = "duplicate_imu";
            else {
              const auto delta = p.imu_ms - kf_ms;
              if (delta > 200) {
                kf_fault = true;
                state = "filter_gap_restart_required";
              } else {
                kalman.update(pose.r, delta * .001, pose.yaw);
                fresh = true;
                state = "updated";
              }
            }
          }
          out << ',' << state;
          if (fresh) {
            kf_ms = p.imu_ms;
            kf_seq = p.imu_sequence;
            out << ',' << kalman.yaw() << ',' << kalman.bias() << ','
                << pose.r - kalman.bias();
          } else
            out << ",,,";
        }
        if (host_timestamps)
          out << ',' << host_read_ns << ','
              << static_cast<double>(host_read_ns - host_origin_ns) * 1e-9
              << ',' << host_batch_id;
        if (diagnostics) {
          out << ',' << p.protocol_version << ',' << p.enc_sequence << ',' << p.imu_sequence
              << ',' << p.delta_ticks << ',' << p.speed << ',' << p.servo_deg << ',' << p.pwm
              << ',' << p.ax << ',' << p.ay << ',' << p.gyro_dps << ',' << p.yaw_deg;
          if(p.protocol_version==2)
            out << ',' << p.gyro_x_dps << ',' << p.gyro_y_dps << ',' << p.roll_deg << ',' << p.pitch_deg
                << ',' << p.read_start_ms << ',' << p.read_end_ms << ',' << p.enc_interval_ms
                << ',' << p.system_status << ',' << p.system_error;
          else out << ",,,,,,,,,";
        }
        out << '\n';
        buffer.clear();
      }
    };
    if (!input.empty()) {
      std::ifstream in(input, std::ios::binary);
      if (!in)
        throw std::runtime_error("cannot open input");
      char bytes[256];
      while (in) {
        in.read(bytes, sizeof(bytes));
        consume(bytes, static_cast<std::size_t>(in.gcount()));
      }
    } else {
      Port port;
      port.fd = ::open(device.c_str(), O_RDONLY | O_NOCTTY | O_NONBLOCK);
      if (port.fd < 0)
        throw std::runtime_error(std::strerror(errno));
      if (tcflush(port.fd, TCIFLUSH) != 0)
        throw std::runtime_error("initial input flush failed");
      termios settings{};
      if (tcgetattr(port.fd, &settings) != 0)
        throw std::runtime_error("tcgetattr failed");
      cfmakeraw(&settings);
      cfsetispeed(&settings, B115200);
      cfsetospeed(&settings, B115200);
      settings.c_cflag =
          (settings.c_cflag & ~(CSIZE | PARENB | CSTOPB | CRTSCTS)) | CS8 |
          CLOCAL | CREAD;
      if (tcsetattr(port.fd, TCSANOW, &settings) != 0)
        throw std::runtime_error("tcsetattr failed");
      std::signal(SIGINT, stop);
      std::signal(SIGTERM, stop);
      using Clock = std::chrono::steady_clock;
      const auto startup_start = Clock::now();
      Clock::time_point capture_start;
      bool capturing = false, align_to_line = false;
      std::size_t discarded_bytes = 0;
      while (!stopped) {
        const auto now = Clock::now();
        if (!capturing &&
            std::chrono::duration<double, std::milli>(now - startup_start)
                    .count() >= startup_discard_ms) {
          // Adapters may deliver their own old FIFO after the initial host
          // flush.
          if (startup_discard_ms > 0 && tcflush(port.fd, TCIFLUSH) != 0)
            throw std::runtime_error("post-startup input flush failed");
          capture_start = Clock::now();
          capturing = true;
          align_to_line = startup_discard_ms > 0;
          std::cerr << "Serial capture started after " << startup_discard_ms
                    << " ms startup discard; keep the vehicle at its initial "
                       "pose until now\n";
        }
        if (capturing &&
            std::chrono::duration<double>(Clock::now() - capture_start)
                    .count() >= duration)
          break;
        pollfd event{port.fd, POLLIN, 0};
        int result = ::poll(&event, 1, 100);
        if (result < 0) {
          if (errno == EINTR)
            continue;
          throw std::runtime_error("poll failed");
        }
        if (event.revents & (POLLERR | POLLHUP | POLLNVAL))
          throw std::runtime_error("serial disconnected");
        if (event.revents & POLLIN) {
          char bytes[256];
          auto count = ::read(port.fd, bytes, sizeof(bytes));
          if (count > 0) {
            // One timestamp per completed OS read, before raw logging/parsing.
            // All complete frames in this batch share it; not per-byte wire
            // time.
            host_read_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    Clock::now().time_since_epoch())
                    .count());
            ++host_batch_id;
            const auto size = static_cast<std::size_t>(count);
            if (!capturing) {
              discarded_bytes += size;
              continue;
            }
            std::size_t offset = 0;
            if (align_to_line) {
              while (offset < size) {
                char ch = bytes[offset++];
                ++discarded_bytes;
                if (ch == '\n') {
                  align_to_line = false;
                  break;
                }
              }
            }
            if (!align_to_line && offset < size)
              consume(bytes + offset, size - offset);
          } else if (count < 0 && errno != EAGAIN && errno != EINTR)
            throw std::runtime_error("serial read failed");
        }
      }
      // tcflush may drop additional kernel-buffered bytes not included in this
      // count.
      std::cerr << "startup_discarded_read_bytes=" << discarded_bytes << '\n';
    }
    if (!buffer.empty() || discard) {
      ++bad;
      std::cerr << "Incomplete trailing frame ignored\n";
    }
    std::cerr << "CRC-valid=" << good << " invalid_frames=" << bad
              << " odometry_rejected=" << rejected
              << " fault_latched=" << odom.faulted() << '\n';
    if (kalman_enabled)
      std::cerr << "kf_fault_latched=" << kf_fault << '\n';
    return good == 0 || odom.faulted() || kf_fault ? 2 : 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
