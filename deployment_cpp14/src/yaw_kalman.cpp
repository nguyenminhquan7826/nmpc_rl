#include "yaw_kalman.hpp"
#include <cmath>
#include <stdexcept>
namespace deployment {
namespace {
constexpr double rad = 3.14159265358979323846 / 180.;
}
YawKalman::YawKalman(YawKalmanConfig c) : cfg_(c) {
  for (double v : {c.heading_sigma_deg, c.gyro_sigma_dps, c.bias_rw_dps})
    if (!std::isfinite(v) || v <= 0 || v > 1000)
      throw std::runtime_error("KF noise settings must be finite in (0,1000]");
}
void YawKalman::initialize(double yaw) {
  if (!std::isfinite(yaw))
    throw std::runtime_error("invalid KF initial yaw");
  yaw_ = yaw;
  bias_ = 0;
  p_ = {{rad * rad, 0, 0, rad * rad}};
  initialized_ = true;
}
void YawKalman::update(double rate, double dt, double heading,
                       bool heading_valid) {
  if (!initialized_ || !std::isfinite(rate) || !std::isfinite(dt) || dt <= 0 ||
      dt > .2 || (heading_valid && !std::isfinite(heading)))
    throw std::runtime_error(
        "invalid KF sample/gap; segment or restart required");
  const double sg = cfg_.gyro_sigma_dps * rad,
               qb = std::pow(cfg_.bias_rw_dps * rad, 2);
  yaw_ += (rate - bias_) * dt;
  const double a = p_[0] - dt * (p_[1] + p_[2]) + dt * dt * p_[3] +
                   sg * sg * dt * dt + qb * dt * dt * dt / 3;
  const double b = p_[1] - dt * p_[3] - qb * dt * dt / 2;
  const double c = p_[2] - dt * p_[3] - qb * dt * dt / 2;
  const double d = p_[3] + qb * dt;
  p_ = {{a, b, c, d}};
  if (!heading_valid)
    return;
  const double R = std::pow(cfg_.heading_sigma_deg * rad, 2), S = a + R,
               k0 = a / S, k1 = c / S;
  const double innovation =
      std::atan2(std::sin(heading - yaw_), std::cos(heading - yaw_));
  yaw_ += k0 * innovation;
  bias_ += k1 * innovation;
  // Joseph covariance: (I-KH) P (I-KH)' + K R K'.
  const double u = 1 - k0;
  const double p00 = u * u * a + k0 * k0 * R;
  const double p01 = u * (b - k1 * a) + k0 * k1 * R;
  const double p10 = u * (c - k1 * a) + k1 * k0 * R;
  const double p11 = d - k1 * (b + c) + k1 * k1 * a + k1 * k1 * R;
  p_ = {{p00, (p01 + p10) / 2, (p01 + p10) / 2, p11}};
}
} // namespace deployment
