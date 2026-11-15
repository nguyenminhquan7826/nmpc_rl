#pragma once
#include "telemetry.hpp"
#include <string>
namespace deployment {
struct HardwareLimits {
  double max_speed = .15, max_steering = 22.835397 * pi / 180;
  int servo_sign = 1;
};
struct HardwareCommand {
  double speed, steering;
  int servo;
  bool clipped;
};
// Calibration copied from laas-bfmc step12-status-sync SteeringCalibration.hpp.
// Units: bicycle steering rad, absolute integer servo deg, speed m/s.
HardwareCommand
hardware_command(double vx, double acceleration, double steering, double dt,
                 double rolling_acceleration, const HardwareLimits &limits,
                 double steering_lower = -pi, double steering_upper = pi);
double wheel_angle(int servo, int sign = 1);
std::string checked_frame(const std::string &payload);
std::string command_frame(std::uint32_t seq, const std::string &action,
                          double speed, int servo);
struct Ack {
  std::uint32_t seq = 0, ms = 0, last_seq = 0;
  unsigned reason = 0;
  bool armed = false;
};
bool parse_ack(const std::string &line, Ack &ack);
} // namespace deployment
