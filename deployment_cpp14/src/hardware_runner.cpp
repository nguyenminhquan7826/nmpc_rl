#include "hardware_session.hpp"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <sstream>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <termios.h>
#include <thread>
#include <unistd.h>
namespace {
using namespace deployment;
using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t interrupted = 0;
void signal_handler(int) { interrupted = 1; }
double seconds(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}
struct Serial {
  int fd = -1;
  bool drive = false, owned = false;
  std::uint32_t sequence = 0;
  std::ofstream *tx = nullptr;
  ~Serial() {
    if (fd >= 0) {
      if (drive && owned) {
        try {
          send(command_frame(++sequence, "STOP", 0, 75));
        } catch (...) {
        }
      }
      if (owned)
        ::ioctl(fd, TIOCNXCL);
      close(fd);
    }
  }
  void open_device(const std::string &path) {
    fd = ::open(path.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    ensure(fd >= 0, "Cannot open serial: " + std::string(std::strerror(errno)));
    ensure(flock(fd, LOCK_EX | LOCK_NB) == 0, "Serial already owned");
    termios t{};
    ensure(tcgetattr(fd, &t) == 0, "Cannot read serial attributes");
    cfmakeraw(&t);
    cfsetispeed(&t, B115200);
    cfsetospeed(&t, B115200);
    t.c_cflag |= CLOCAL | CREAD;
    t.c_cflag &= ~CSTOPB;
    t.c_cflag &= ~CRTSCTS;
    ensure(tcsetattr(fd, TCSANOW, &t) == 0, "Cannot configure serial");
    ensure(::ioctl(fd, TIOCEXCL) == 0, "Cannot reserve serial");
    owned = true;
  }
  void send(const std::string &line) {
    ensure(drive && owned && fd >= 0, "Unowned/shadow serial cannot transmit");
    std::size_t n = 0;
    auto start = Clock::now();
    while (n < line.size()) {
      auto k = ::write(fd, line.data() + n, line.size() - n);
      if (k > 0)
        n += static_cast<std::size_t>(k);
      else {
        ensure(k < 0 && (errno == EAGAIN || errno == EINTR),
               "Serial write failed");
        ensure(seconds(start) < .02, "Serial write deadline");
        pollfd p{fd, POLLOUT, 0};
        ::poll(&p, 1, 1);
      }
    }
    if (tx) {
      *tx << line;
      tx->flush();
      ensure(bool(*tx), "TX log write failed");
    }
  }
};
class Keyboard {
  termios original_{};
  bool active_ = false;

public:
  explicit Keyboard(bool enabled) {
    if (!enabled)
      return;
    ensure(tcgetattr(STDIN_FILENO, &original_) == 0,
           "--keyboard requires an interactive terminal");
    auto mode = original_;
    mode.c_lflag &= ~(ICANON | ECHO);
    mode.c_cc[VMIN] = 0;
    mode.c_cc[VTIME] = 0;
    ensure(tcsetattr(STDIN_FILENO, TCSANOW, &mode) == 0,
           "Cannot configure keyboard terminal");
    active_ = true;
  }
  ~Keyboard() {
    if (active_)
      tcsetattr(STDIN_FILENO, TCSANOW, &original_);
  }

  bool stop_requested(bool &start_requested) const {
    if (!active_)
      return false;
    pollfd input{STDIN_FILENO, POLLIN, 0};
    const int ready = ::poll(&input, 1, 0);
    if (ready < 0 && errno == EINTR)
      return false;
    ensure(ready >= 0, "Keyboard poll failed");
    if (input.revents & (POLLHUP | POLLERR | POLLNVAL))
      return true;
    if (!(input.revents & POLLIN))
      return false;
    char keys[64];
    const auto count = ::read(STDIN_FILENO, keys, sizeof(keys));
    if (count < 0 && errno == EINTR)
      return false;
    ensure(count >= 0, "Keyboard read failed");
    for (ssize_t i = 0; i < count; ++i) {
      if (keys[i] == 'q')
        return true;
      if (keys[i] == 'r')
        start_requested = true;
    }
    return false;
  }
};

struct Options {
  std::string model, settings, reference, output, device, input, mode = "fixed";
  bool drive = false, keyboard = false, x = false, y = false, yaw = false;
  double x0 = 0, y0 = 0, yaw0 = 0, duration = 60, discard_ms = 1000;
  unsigned seed = 0;
};
Options options(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--drive") {
      o.drive = true;
      continue;
    }
    if (a == "--keyboard") {
      o.keyboard = true;
      continue;
    }
    ensure(i + 1 < argc, "Missing option value");
    std::string v = argv[++i];
    if (a == "--model")
      o.model = v;
    else if (a == "--settings")
      o.settings = v;
    else if (a == "--reference")
      o.reference = v;
    else if (a == "--output")
      o.output = v;
    else if (a == "--device")
      o.device = v;
    else if (a == "--input")
      o.input = v;
    else if (a == "--mode")
      o.mode = v;
    else {
      std::size_t used = 0;
      double n = std::stod(v, &used);
      ensure(used == v.size() && std::isfinite(n), "Invalid numeric argument");
      if (a == "--x0") {
        o.x0 = n;
        o.x = true;
      } else if (a == "--y0") {
        o.y0 = n;
        o.y = true;
      } else if (a == "--yaw0-deg") {
        o.yaw0 = n * pi / 180;
        o.yaw = true;
      } else if (a == "--duration")
        o.duration = n;
      else if (a == "--startup-discard-ms")
        o.discard_ms = n;
      else if (a == "--seed") {
        ensure(n >= 0 && n <= 2147483647 && std::floor(n) == n, "Invalid seed");
        o.seed = static_cast<unsigned>(n);
      } else
        throw std::invalid_argument("Unknown option: " + a);
    }
  }
  const bool shorthand = o.model.empty() || o.settings.empty() ||
      o.reference.empty() || o.output.empty() ||
      (o.device.empty() && o.input.empty());
  if (o.model.empty()) o.model = "hardware_trial_inputs/initial.model";
  if (o.settings.empty()) o.settings = "hardware_trial_inputs/hardware.settings";
  if (o.reference.empty()) o.reference = "hardware_trial_inputs/reference.csv";
  if (o.device.empty() && o.input.empty()) o.device = "/dev/ttyACM0";
  // Preserve explicit legacy/PTY CLI; shorthand drive always waits for r.
  if (o.drive && shorthand) o.keyboard = true;
  ensure(o.device.empty() != o.input.empty(),
         "Choose exactly one device or replay input");
  ensure(!o.drive || (!o.device.empty() && o.x && o.y && o.yaw),
         "Drive requires serial and explicit x0/y0/yaw0-deg");
  ensure(!o.keyboard || o.drive, "--keyboard requires --drive");
  ensure(o.duration > 0 && o.duration <= 3600 && o.discard_ms >= 0 &&
             o.discard_ms <= 10000,
         "Invalid duration/discard");
  return o;
}
std::string default_output(const Options &o) {
  const std::string parent = "hardware_runs";
  if (::mkdir(parent.c_str(), 0755) != 0) {
    ensure(errno == EEXIST, "Cannot create hardware_runs");
    struct stat st{};
    ensure(::stat(parent.c_str(), &st) == 0 && S_ISDIR(st.st_mode),
           "hardware_runs is not a directory");
  }
  const auto stamp = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  return parent + "/" + (o.drive ? "drive_" : o.input.empty() ? "shadow_" : "replay_")
      + o.mode + "_" + std::to_string(stamp) + "_" + std::to_string(::getpid());
}
void copy_file(const std::string &a, const std::string &b) {
  std::ifstream in(a, std::ios::binary);
  std::ofstream out(b, std::ios::binary);
  ensure(bool(in) && bool(out), "Cannot snapshot inputs");
  out << in.rdbuf();
  ensure(bool(out), "Snapshot write failed");
}
} // namespace
int main(int argc, char **argv) {
  using namespace deployment;
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      std::cout << "Run from deployment_cpp14 after prepare.\n"
                   "hardware_runner --mode fixed|actor_adapt|critic_adapt|full_adapt\n"
                   "Defaults: hardware_trial_inputs/{initial.model,hardware.settings,reference.csv},\n"
                   "/dev/ttyACM0, duration 60s, seed 0, new hardware_runs directory.\n"
                   "Options: --model --settings --reference --output --device --input\n"
                   "         --duration --seed --startup-discard-ms\n"
                   "Shadow by default. Drive requires --drive and explicit\n"
                   "--x0 X --y0 Y --yaw0-deg DEG; shorthand drive waits for r, q stops.\n"
                   "Use --keyboard for explicit drive. No replay drive.\n";
      return 0;
    }
    auto o = options(argc, argv);
    std::map<std::string, Mode> modes{{"fixed", Mode::Fixed},
                                      {"actor_adapt", Mode::ActorAdapt},
                                      {"critic_adapt", Mode::CriticAdapt},
                                      {"full_adapt", Mode::FullAdapt}};
    ensure(modes.count(o.mode), "Invalid mode");
    auto model = read_numbers(o.model, "AC_MODEL_V1"), // đọc dữ liệu số của  các thông số ator và critic
         config = read_numbers(o.settings, "NMPC_HARDWARE_V1"); // lấy các thông số cuẩ bộ điều khienr mpc
    auto path = ReferencePath::load_csv(o.reference); // đọc quỹ đạo tham chiếu
    /*
    khởi tạo các thông số actor-critic
    cấu hình nmpc và phần cứng quỹ đạo tham chiếu
    chế độ điều khiển
    seed cho bộ sinh số ngẫu nhiên
    */
    HardwareSession session(model, config, path, modes.at(o.mode), o.seed);
    // tính toán thời gian và chi phí tính toán của các phần => đánh giá được đáp ứng thời gian thực không
    const double timeout = scalar(config, "state_timeout_ms") * .001,
                 budget = scalar(config, "solve_budget_ms"),
                 ack_timeout = scalar(config, "ack_timeout_ms") * .001;
    ensure(timeout >= .04 && timeout <= .15 && budget > 0 &&
               budget < session.period() * 1000 && ack_timeout > 0 &&
               ack_timeout <= .15,
           "Invalid runtime deadlines");
    
    Config odom_cfg;
    odom_cfg.gyro_sign = 1;
    odom_cfg.max_speed = scalar(config, "max_speed") + .05;
    odom_cfg.x0 = o.x ? o.x0 : path.rows().front().state[4];
    odom_cfg.y0 = o.y ? o.y0 : path.rows().front().state[5];
    odom_cfg.yaw0 = o.yaw ? o.yaw0 : path.rows().front().state[2];
    Odometry odom(odom_cfg);
    
    if (o.output.empty()) o.output = default_output(o);
    ensure(::mkdir(o.output.c_str(), 0755) == 0,
           "Output directory exists or cannot be created");
    std::cerr << "Output: " << o.output << '\n';
    copy_file(o.model, o.output + "/initial.model");
    copy_file(o.settings, o.output + "/settings.txt");
    copy_file(o.reference, o.output + "/reference.csv");
    std::ofstream transitions(o.output + "/transitions.csv");
    transitions << std::setprecision(17)
                << "host_s,command_seq,imu_before_ms,imu_after_ms,ack_ms,"
                   "reward,td,actor_step_norm,critic_step_norm,batch_size";
    for (auto name : {"error", "next_error"})
      for (int j = 0; j < 6; j++)
        transitions << ',' << name << j;
    transitions << ",feedback_ms,observe_total_ms,actor_update_ms,critic_evaluate_ms,critic_update_ms,run_ack_wait_ms\n";
    std::ofstream raw(o.output + "/rx.log", std::ios::binary),
        tx(o.output + "/tx.log", std::ios::binary),
        events(o.output + "/events.csv"), log(o.output + "/control.csv"),
        states(o.output + "/state.csv");
    ensure(bool(raw) && bool(tx) && bool(events) && bool(log) && bool(states),
           "Cannot create logs");
    const auto origin = Clock::now();
    const auto origin_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               origin.time_since_epoch())
                               .count();
    events << "host_s,event,sequence,detail\n";
    log << std::setprecision(17)
        << "host_s,step,mode,imu_ms,X,Y,yaw,vx,vy_assumed,r,progress,cte,solve_"
           "ms,solver_success,clipped,speed_request,servo_request,a_nmpc,delta_"
           "nmpc,delta_quantized,actor_updates,critic_updates";
    for (auto k :
         {"Q", "R", "P", "obs", "mu", "sigma", "latent", "reference"}) {
      for (int j = 0;
           j < (std::string(k) == "R"     ? 2
                : std::string(k) == "obs" ? 17
                : (std::string(k) == "mu" || std::string(k) == "sigma" ||
                   std::string(k) == "latent")
                    ? 8
                    : 6);
           j++)
        log << ',' << k << j;
    }
    log << ",slot_host_s,period_ms,reference_ms,feature_ms,actor_proposal_ms,warm_start_ms,command_mapping_ms,proposal_total_ms,feedback_ms,control_compute_ms\n";
    states << std::setprecision(17)
           << "host_s,packet_seq,tx_ms,imu_ms,accepted,status,X,Y,yaw,vx,r,"
              "system_status,system_error\n";
    auto event = [&](const std::string &kind, std::uint32_t seq,
                     const std::string &detail) {
      std::string safe = detail.substr(0, 512);
      for (char &c : safe)
        if (c == ',' || c == '\n' || c == '\r')
          c = ' ';
      events << std::setprecision(17) << seconds(origin) << ',' << kind << ','
             << seq << ',' << safe << '\n';
      events.flush();
      ensure(bool(events), "Event log failed");
    };
    Serial port;
    port.drive = o.drive;
    port.tx = &tx;
    std::ifstream replay;
    if (!o.device.empty()) {
      port.open_device(o.device);
      char bytes[2048];
      auto start = Clock::now();
      while (seconds(start) < o.discard_ms * .001) {
        ::read(port.fd, bytes, sizeof(bytes));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      tcflush(port.fd, TCIFLUSH);
    } else {
      replay.open(o.input, std::ios::binary);
      ensure(bool(replay), "Cannot open replay input");
    }
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    Keyboard keyboard(o.keyboard);
    bool start_requested = !o.keyboard;
    if (o.keyboard)
      std::cerr << "Press r to start, q to stop and exit.\n";
    enum Phase { SyncStop, WaitHealthy, ArmAck, Ready, RunAck, Feedback };
    Phase phase = o.drive ? SyncStop : Ready;
    std::uint32_t expected = 0, last_imu = 0, ack_ms = 0;
    std::size_t step = 0, valid = 0, rejected = 0;
    bool have = false, have_step = false;
    Telemetry latest{};
    Pose pose;
    HardwareStep current;
    auto last_host = Clock::now(), sent_at = Clock::now();
    // Both live modes use phase-locked host time. Drive fails closed if
    // ACK or fresh feedback cannot meet the slot; replay uses MCU time.
    const bool live_shadow = !o.drive && !o.device.empty();
    const auto shadow_period = std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(session.period()));
    auto shadow_next = Clock::time_point{};
    std::size_t shadow_missed_slots = 0;
    double shadow_lateness_max_ms = 0;
    auto drive_next = Clock::time_point{};
    // Conservative experimental bounds, not a hardware timing guarantee.
    const auto drive_grace = std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(std::min(.010, session.period() / 8)));
    const double feedback_tolerance = std::min(.015, session.period() * .2);
    std::size_t drive_deadline_misses = 0, feedback_count = 0;
    double drive_lateness_max_ms = 0, feedback_sum_ms = 0,
           feedback_max_ms = 0, drive_compute_max_ms = 0;
    std::vector<double> solve_times;
    std::vector<double> period_times, compute_times, actor_times, reference_times,
        feature_times, mapping_times, proposal_times, warm_start_times, ack_wait_times,
        actor_update_times, critic_evaluate_times, critic_update_times, observe_times;
    auto previous_slot = Clock::time_point{};
    double run_ack_wait_ms = 0;
    auto elapsed_ms = [](Clock::time_point t) {
      return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
    };
    std::string reason = "completed", buffer;
    bool line_overflow = false;
    int exit_code = 0;
    auto transmit = [&](const std::string &action, double speed, int servo) {
      expected = ++port.sequence;
      port.send(command_frame(expected, action, speed, servo));
      sent_at = Clock::now();
      event("tx_" + action, expected, "requested");
    };
    if (o.drive) {
      port.sequence = UINT32_MAX;
      transmit("STOP", 0, 75);
    }
    auto stop = [&](const std::string &why) {
      reason = why;
      exit_code = 2;
      event("fault", latest.sequence, why);
      session.cancel();
    };
    auto consume = [&](const std::string &line) {
      Ack ack;
      if (line.compare(0, 4, "ACK,") == 0) {
        if (!parse_ack(line, ack)) {
          event("bad_ack", 0, "crc_or_format");
          return;
        }
        event("ack", ack.seq,
              std::to_string(ack.reason) + "_armed_" +
                  std::to_string(ack.armed));
        if (!o.drive || ack.seq != expected)
          return;
        if (phase != SyncStop && phase != ArmAck && phase != RunAck)
          return;
        if (seconds(sent_at) > ack_timeout) {
          stop("ack_timeout");
          return;
        }
        if (ack.reason != 0) {
          stop("command_rejected");
          return;
        }
        if (phase == SyncStop) {
          if (ack.armed) {
            stop("stop_not_confirmed");
            return;
          }
          port.sequence = ack.last_seq;
          phase = WaitHealthy;
        } else if (phase == ArmAck) {
          if (!ack.armed) {
            stop("arm_not_confirmed");
            return;
          }
          phase = Ready;
        } else if (phase == RunAck) {
          if (!ack.armed) {
            stop("run_not_confirmed");
            return;
          }
          run_ack_wait_ms = elapsed_ms(sent_at);
          ack_wait_times.push_back(run_ack_wait_ms);
          session.accept();
          ack_ms = ack.ms;
          phase = Feedback;
        }
        return;
      }
      Telemetry p{};
      std::string status;
      if (!parse(line, p, status)) {
        event("invalid_frame", 0, status);
        return;
      }
      if (p.protocol_version != 2) {
        stop("TEL2_required");
        return;
      }
      bool accepted = odom.update(p, status);
      const auto &s = odom.pose();
      states << seconds(origin) << ',' << p.sequence << ',' << p.tx_ms << ','
             << p.imu_ms << ',' << accepted << ',' << status << ',' << s.x
             << ',' << s.y << ',' << s.yaw << ',' << s.vx << ',' << s.r << ','
             << p.system_status << ',' << p.system_error << '\n';
      states.flush();
      ensure(bool(states), "State log failed");
      if (!accepted) {
        rejected++;
        if (odom.faulted())
          stop("odometry_fault");
        return;
      }
      if (p.system_status != 5 || p.system_error != 0) {
        stop("fusion_unhealthy");
        return;
      }
      if (have &&
          (p.imu_ms == latest.imu_ms || p.imu_sequence == latest.imu_sequence))
        return;
      latest = p;
      pose = s;
      have = true;
      valid++;
      last_host = Clock::now();
    };
    auto check_keyboard = [&]() {
      if (keyboard.stop_requested(start_requested)) {
        reason = "operator_stop";
        return true;
      }
      return interrupted != 0;
    };
    auto tick = [&]() {
      if (check_keyboard())
        return;
      if (o.drive &&
          (phase == SyncStop || phase == ArmAck || phase == RunAck) &&
          seconds(sent_at) > ack_timeout) {
        stop("ack_timeout");
        return;
      }
      if (!o.device.empty() && seconds(last_host) > timeout) {
        stop(have ? "telemetry_timeout" : "no_healthy_telemetry");
        return;
      }
      if (!have)
        return;
      // Check even while waiting for RUN ACK, not just when feedback arrives.
      if (o.drive && have_step && Clock::now() > drive_next + drive_grace) {
        ++drive_deadline_misses;
        stop("drive_slot_deadline");
        return;
      }
      if (phase == WaitHealthy) {
        if (!start_requested)
          return;
        if (std::abs(pose.vx) > .02 || std::abs(pose.r) > .1) {
          stop("arm_requires_stationary_vehicle");
          return;
        }
        transmit("ARM", 0, 75);
        phase = ArmAck;
        return;
      }
      if (phase != Ready && phase != Feedback)
        return;
      if (have_step && latest.imu_ms == last_imu)
        return;
      if (live_shadow && have_step) {
        // Never require MCU timestamps to differ by exactly Ts: 79 ms of
        // sensor progress must not force a further 40 ms wait on the host.
        if (Clock::now() < shadow_next)
          return;
      }
      if (have_step && o.drive) {
        if (Clock::now() < drive_next)
          return;
        const double delta =
            static_cast<std::uint32_t>(latest.imu_ms - last_imu) * .001;
        // MCU elapsed time is a feedback validity check, not the scheduler.
        // A 78/79 ms sample is usable; a 40/120 ms transition is not.
        if (delta + 1e-6 < session.period() - feedback_tolerance)
          return;
        if (delta > session.period() + feedback_tolerance + 1e-6) {
          stop("drive_feedback_interval");
          return;
        }
      }
      if (have_step && !live_shadow && !o.drive) {
        const auto delta =
            static_cast<std::uint32_t>(latest.imu_ms - last_imu) * .001;
        if (delta + 1e-6 < session.period())
          return;
      }
      const auto slot_begin = Clock::now();
      const auto slot_due = o.drive && have_step ? drive_next : slot_begin;
      if (o.drive && have_step && slot_begin > slot_due + drive_grace) {
        ++drive_deadline_misses;
        stop("drive_slot_deadline");
        return;
      }
      double feedback_this_ms = 0;
      if (phase == Feedback) {
        auto after = static_cast<std::uint32_t>(latest.read_start_ms - ack_ms);
        if (after == 0 || after >= 0x80000000U)
          return;
        bool truncating = seconds(origin) + session.period() >= o.duration;
        const auto feedback_begin = Clock::now();
        auto update = session.feedback(pose, false, truncating);
        const double feedback_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - feedback_begin).count();
        feedback_this_ms = feedback_ms;
        observe_times.push_back(update.observe_total_ms);
        critic_evaluate_times.push_back(update.critic_evaluate_ms);
        if (update.batch_size) actor_update_times.push_back(update.actor_update_ms);
        if (o.mode == "critic_adapt" || o.mode == "full_adapt")
          critic_update_times.push_back(update.critic_update_ms);
        ++feedback_count;
        feedback_sum_ms += feedback_ms;
        feedback_max_ms = std::max(feedback_max_ms, feedback_ms);
        transitions << seconds(origin) << ',' << expected << ',' << last_imu
                    << ',' << latest.imu_ms << ',' << ack_ms << ','
                    << session.reward() << ',' << update.td << ','
                    << update.actor_step_norm << ',' << update.critic_step_norm
                    << ',' << update.batch_size;
        for (const auto &v : {current.error, session.next_error()})
          for (double x : v)
            transitions << ',' << x;
        transitions << ',' << feedback_ms << ',' << update.observe_total_ms
                    << ',' << update.actor_update_ms << ',' << update.critic_evaluate_ms
                    << ',' << update.critic_update_ms << ',' << run_ack_wait_ms << '\n';
        transitions.flush();
        ensure(bool(transitions), "Transition log failed");
        event("learning", latest.sequence,
              "td_" + std::to_string(update.td) + "_actor_step_" +
                  std::to_string(update.actor_step_norm) + "_critic_step_" +
                  std::to_string(update.critic_step_norm));
        phase = Ready;
        if (session.reached() || truncating) {
          reason = session.reached() ? "route_completed" : "duration_completed";
          exit_code = 0;
          return;
        }
      }
      auto start = Clock::now();
      if (o.drive) {
        drive_lateness_max_ms = std::max(
            drive_lateness_max_ms,
            std::chrono::duration<double, std::milli>(slot_begin - slot_due).count());
        drive_next = slot_due + shadow_period;
      }
      if (live_shadow) {
        if (!have_step) {
          shadow_next = start + shadow_period;
        } else {
          const double late_ms =
              std::chrono::duration<double, std::milli>(start - shadow_next).count();
          shadow_lateness_max_ms = std::max(shadow_lateness_max_ms, late_ms);
          shadow_next += shadow_period;
          // Skip overdue slots (and slots too close to a late execution).
          // Never execute multiple steps in a burst to catch up.
          while (shadow_next <= start + shadow_period / 2) {
            shadow_next += shadow_period;
            ++shadow_missed_slots;
          }
        }
      }
      current = session.propose(pose);
      const double proposal_ms = elapsed_ms(start);
      const double compute_ms = elapsed_ms(slot_begin);
      const double slot_host_s = std::chrono::duration<double>(slot_begin - origin).count();
      const bool has_period = previous_slot != Clock::time_point{};
      const double period_ms = has_period
          ? std::chrono::duration<double, std::milli>(slot_begin - previous_slot).count() : 0;
      previous_slot = slot_begin;
      if (has_period && !o.device.empty()) period_times.push_back(period_ms);
      compute_times.push_back(compute_ms);
      actor_times.push_back(current.actor_proposal_ms);
      reference_times.push_back(current.reference_ms);
      feature_times.push_back(current.feature_ms);
      mapping_times.push_back(current.command_mapping_ms);
      proposal_times.push_back(proposal_ms);
      warm_start_times.push_back(current.warm_start_ms);
      solve_times.push_back(current.solve_ms);
      step++;
      last_imu = latest.imu_ms;
      have_step = true;
      log << seconds(origin) << ',' << step << ',' << o.mode << ','
          << latest.imu_ms << ',' << pose.x << ',' << pose.y << ',' << pose.yaw
          << ',' << pose.vx << ",0," << pose.r << ',' << current.progress << ','
          << current.cte << ',' << current.solve_ms << ',' << current.success
          << ',' << current.command.clipped << ',' << current.command.speed
          << ',' << current.command.servo;
      if (current.success)
        log << ',' << current.control[0] << ',' << current.control[1] << ','
            << current.command.steering;
      else
        log << ",,,";
      log << ',' << session.actor_updates() << ',' << session.critic_updates();
      for (const auto &v :
           {current.q, current.r, current.p, current.observation, current.mu,
            current.sigma, current.latent, current.reference})
        for (double x : v)
          log << ',' << x;
      log << ',' << slot_host_s << ',';
      if (has_period && !o.device.empty()) log << period_ms;
      log << ',' << current.reference_ms << ',' << current.feature_ms
          << ',' << current.actor_proposal_ms << ',' << current.warm_start_ms
          << ',' << current.command_mapping_ms << ',' << proposal_ms
          << ',' << feedback_this_ms << ',' << compute_ms << '\n';
      log.flush();
      ensure(bool(log), "Control log failed");
      if (current.reached) {
        session.cancel();
        reason = "route_completed";
        exit_code = 0;
        return;
      }
      if (!current.success) {
        stop("solver_failure");
        return;
      }
      if (current.solve_ms > budget || seconds(last_host) > timeout) {
        stop("solve_deadline_or_stale_state");
        return;
      }
      if (o.drive) {
        if (current.command.clipped) {
          stop("command_clipped");
          return;
        }
        // Include feedback, learning, proposal and logging in the budget.
        // Reserve the existing serial write bound before starting RUN.
        drive_compute_max_ms = std::max(
            drive_compute_max_ms,
            std::chrono::duration<double, std::milli>(Clock::now() - slot_begin).count());
        if (Clock::now() + std::chrono::milliseconds(20) >= drive_next) {
          ++drive_deadline_misses;
          stop("loop_deadline");
          return;
        }
        if (check_keyboard())
          return;
        if (seconds(origin) >= o.duration) {
          session.cancel();
          reason = "duration_completed";
          return;
        }
        transmit("RUN", current.command.speed, current.command.servo);
        if (Clock::now() >= drive_next) {
          ++drive_deadline_misses;
          stop("run_write_deadline");
          return;
        }
        phase = RunAck;
      } else {
        session.cancel();
        event("shadow", latest.sequence, "not_sent_not_learned");
      }
    };
    try {
      while (!interrupted && reason == "completed" &&
             seconds(origin) < o.duration) {
        if (port.fd >= 0) {
          pollfd p{port.fd, POLLIN, 0};
          int ready = ::poll(&p, 1, 2);
          ensure(ready >= 0 || errno == EINTR, "Serial poll failed");
          if (ready > 0) {
            ensure(!(p.revents & (POLLERR | POLLHUP | POLLNVAL)),
                   "Serial disconnected");
            char bytes[4096];
            auto count = ::read(port.fd, bytes, sizeof(bytes));
            if (count > 0) {
              raw.write(bytes, count);
              raw.flush();
              ensure(bool(raw), "RX log failed");
              for (ssize_t i = 0; i < count; i++) {
                char c = bytes[i];
                if (c == '\n') {
                  if (!line_overflow && reason == "completed")
                    consume(buffer);
                  buffer.clear();
                  line_overflow = false;
                } else if (c != '\r' && !line_overflow) {
                  if (buffer.size() < 512)
                    buffer += c;
                  else {
                    buffer.clear();
                    line_overflow = true;
                    event("oversize", 0, "discard_until_newline");
                  }
                }
              }
            } else
              ensure(count < 0 && (errno == EAGAIN || errno == EINTR),
                     "Serial read failed");
          }
          if (reason == "completed")
            tick();
        } else {
          std::string line;
          if (!std::getline(replay, line))
            break;
          raw << line << '\n';
          if (line.size() > 512) {
            event("oversize", 0, "replay_skipped");
            continue;
          }
          consume(line);
          if (reason == "completed")
            tick();
        }
      }
      if (interrupted) {
        reason = "interrupted";
        exit_code = 2;
      }
      if (!have && reason == "completed") {
        reason = "no_healthy_telemetry";
        exit_code = 2;
      }
    } catch (const std::exception &e) {
      reason = "runtime_error";
      exit_code = 2;
      event("exception", latest.sequence, e.what());
    }
    session.cancel();
    if (o.drive) {
      try {
        transmit("STOP", 0, 75);
      } catch (const std::exception &e) {
        event("stop_write_failed", 0, e.what());
        exit_code = 2;
      }
    }
    write_numbers(o.output + "/final.model", "AC_MODEL_V1", session.model());
    std::ofstream summary(o.output + "/summary.txt");
    summary << "host_origin_steady_ns=" << origin_ns << "\nmode=" << o.mode
            << "\ndrive=" << o.drive << "\nseed=" << o.seed
            << "\nreason=" << reason << "\nvalid_samples=" << valid
            << "\nrejected_samples=" << rejected << "\ncontrol_steps=" << step
            << "\nactor_updates=" << session.actor_updates()
            << "\ncritic_updates=" << session.critic_updates()
            << "\nvy_assumed_zero=1\nkalman_used=0\n"
            << "scheduler=" << (o.drive ? "host_periodic_drive" :
                                 live_shadow ? "host_periodic_shadow" : "imu_sample_gated")
            << '\n';
    if (o.drive) {
      summary << "drive_deadline_misses=" << drive_deadline_misses
              << "\ndrive_lateness_max_ms=" << drive_lateness_max_ms
              << "\ndrive_compute_max_ms=" << drive_compute_max_ms
              << "\nfeedback_count=" << feedback_count
              << "\nfeedback_mean_ms="
              << (feedback_count ? feedback_sum_ms / feedback_count : 0)
              << "\nfeedback_max_ms=" << feedback_max_ms << '\n';
    }
    if (live_shadow) {
      summary << "shadow_missed_slots=" << shadow_missed_slots
              << "\nshadow_lateness_max_ms=" << shadow_lateness_max_ms << '\n';
    }
    if (!solve_times.empty()) {
      std::sort(solve_times.begin(), solve_times.end());
      double sum = 0;
      for (double t : solve_times)
        sum += t;
      summary << "solve_mean_ms=" << sum / solve_times.size()
              << "\nsolve_p95_ms="
              << solve_times[static_cast<std::size_t>(.95 *
                                                      (solve_times.size() - 1))]
              << "\nsolve_max_ms=" << solve_times.back() << '\n';
    }
    auto profile = [&](const std::string &name, std::vector<double> &v) {
      if (v.empty()) return;
      double sum = 0;
      for (double x : v) sum += x;
      std::sort(v.begin(), v.end());
      summary << name << "_count=" << v.size() << '\n'
              << name << "_mean_ms=" << sum / v.size() << '\n'
              << name << "_p95_ms=" << v[static_cast<std::size_t>(.95 * (v.size()-1))] << '\n'
              << name << "_max_ms=" << v.back() << '\n';
    };
    profile("period", period_times);
    profile("control_compute", compute_times);
    profile("reference", reference_times);
    profile("feature", feature_times);
    profile("actor_proposal", actor_times);
    profile("warm_start", warm_start_times);
    profile("command_mapping", mapping_times);
    profile("proposal_total", proposal_times);
    profile("run_ack_wait", ack_wait_times);
    profile("observe_total", observe_times);
    profile("actor_update", actor_update_times);
    profile("critic_evaluate", critic_evaluate_times);
    profile("critic_update", critic_update_times);
    summary.flush();
    ensure(bool(summary) && bool(raw) && bool(states),
           "Final log write failed");
    std::cerr << "Hardware runner: " << reason << " steps=" << step
              << " drive=" << o.drive << '\n';
    return exit_code;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
