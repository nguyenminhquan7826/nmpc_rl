#include "hardware_control.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>
namespace deployment {
namespace {
void check(bool v, const char *s) {
  if (!v)
    throw std::invalid_argument(s);
}
double poly(double p) { return .9666368131 * p - .0002282854502 * p * p * p; }
std::uint32_t uint_value(const std::string &s) {
  check(!s.empty() && s.find_first_not_of("0123456789") == std::string::npos,
        "Invalid ACK integer");
  auto x = std::stoull(s);
  check(x <= UINT32_MAX, "ACK overflow");
  return static_cast<std::uint32_t>(x);
}
} // namespace
double wheel_angle(int servo, int sign) {
  check(servo >= 45 && servo <= 105 && (sign == 1 || sign == -1),
        "Invalid servo");
  double d = poly(servo - 75);
  return sign * std::copysign(std::min(std::abs(d), 22.835397), d) * pi / 180;
}
HardwareCommand hardware_command(double vx, double a, double delta, double dt,
                                 double drag, const HardwareLimits &l,
                                 double steering_lower, double steering_upper) {
  for (double v : {vx, a, delta, dt, drag, l.max_speed, l.max_steering,
                   steering_lower, steering_upper})
    check(std::isfinite(v), "Nonfinite command");
  check(steering_lower <= steering_upper, "Invalid steering interval");
  check(dt > 0 && dt <= .1 && l.max_speed > 0 && l.max_speed <= .5 &&
            l.max_steering > 0 && l.max_steering <= 22.835397 * pi / 180 &&
            (l.servo_sign == 1 || l.servo_sign == -1),
        "Invalid hardware limits");
  const double wanted = vx + (a - drag) * dt;
  double speed = std::max(0., std::min(l.max_speed, wanted));
  double d = std::max(-l.max_steering, std::min(l.max_steering, delta));
  int servo = 75;
  double best = 1e9;
  bool feasible = false;
  // Select only feasible integer commands; nearest rounding alone can exceed
  // a reduced physical limit (e.g. 5.5 degrees -> servo offset6).
  for (int candidate = 45; candidate <= 105; candidate++) {
    double actual = wheel_angle(candidate, l.servo_sign);
    if (std::abs(actual) > l.max_steering + 1e-12 ||
        actual < steering_lower - 1e-10 || actual > steering_upper + 1e-10)
      continue;
    double distance = std::abs(actual - d);
    if (distance < best) {
      best = distance;
      servo = candidate;
      feasible = true;
    }
  }
  check(feasible, "No integer servo command within absolute/rate limits");
  // Wire speed has six decimals. Log exactly the request sent to firmware.
  speed = std::round(speed * 1e6) / 1e6;
  if (speed > l.max_speed)
    speed = std::floor(l.max_speed * 1e6) / 1e6;
  return {speed, wheel_angle(servo, l.servo_sign), servo,
          std::abs(speed - wanted) > 1e-6 || std::abs(d - delta) > 1e-10};
}
std::string checked_frame(const std::string &payload) {
  std::ostringstream s;
  s << payload << ',' << std::uppercase << std::hex << std::setw(4)
    << std::setfill('0') << crc16(payload) << "\r\n";
  return s.str();
}
std::string command_frame(std::uint32_t seq, const std::string &action,
                          double speed, int servo) {
  check(action == "ARM" || action == "RUN" || action == "STOP",
        "Invalid action");
  check(std::isfinite(speed) && speed >= 0 && speed <= .5 && servo >= 45 &&
            servo <= 105,
        "Command outside firmware limits");
  check(action == "RUN" || (speed == 0 && servo == 75),
        "ARM/STOP require neutral");
  std::ostringstream s;
  s << "CTL,1," << seq << ',' << action << ',' << std::fixed
    << std::setprecision(6) << speed << ',' << servo;
  return checked_frame(s.str());
}
bool parse_ack(const std::string &line, Ack &out) {
  try {
    std::string s = line;
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r'))
      s.pop_back();
    const auto pos = s.rfind(',');
    if (pos == std::string::npos || s.size() - pos != 5)
      return false;
    std::string hex = s.substr(pos + 1);
    if (hex.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos ||
        std::stoul(hex, nullptr, 16) != crc16(s.substr(0, pos)))
      return false;
    std::vector<std::string> v;
    std::istringstream stream(s.substr(0, pos));
    std::string x;
    while (std::getline(stream, x, ','))
      v.push_back(x);
    if (v.size() != 7 || v[0] != "ACK" || v[1] != "1")
      return false;
    Ack a;
    a.seq = uint_value(v[2]);
    auto armed = uint_value(v[3]);
    a.reason = uint_value(v[4]);
    a.ms = uint_value(v[5]);
    a.last_seq = uint_value(v[6]);
    if (armed > 1 || a.reason > 6)
      return false;
    a.armed = armed != 0;
    out = a;
    return true;
  } catch (...) {
    return false;
  }
}
} // namespace deployment
