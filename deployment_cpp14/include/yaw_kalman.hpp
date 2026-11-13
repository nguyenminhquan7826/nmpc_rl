#pragma once
#include <array>
namespace deployment {
struct YawKalmanConfig {
  double heading_sigma_deg = 1., gyro_sigma_dps = .35, bias_rw_dps = .01;
};
// Experimental yaw/bias estimator. Inputs must share a vehicle-frame sign.
// BNO heading/gyro correlation is not modelled; covariance is not calibrated.
class YawKalman {
  YawKalmanConfig cfg_;
  double yaw_ = 0, bias_ = 0;
  std::array<double, 4> p_{};
  bool initialized_ = false;

public:
  explicit YawKalman(YawKalmanConfig config = YawKalmanConfig());
  void initialize(double yaw);
  void update(double rate, double dt, double heading,
              bool heading_valid = true);
  double yaw() const { return yaw_; }
  double bias() const { return bias_; }
  const std::array<double, 4> &covariance() const { return p_; }
};
} // namespace deployment
