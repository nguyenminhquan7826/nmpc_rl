#include "yaw_kalman.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace deployment;
void check(bool c) {
  if (!c)
    throw std::runtime_error("yaw KF test failed");
}
int main() {
  constexpr double rad = 3.14159265358979323846 / 180.;
  YawKalman f;
  f.initialize(359 * rad);
  for (int k = 1; k <= 200; ++k) {
    f.update(5 * rad, .02, std::fmod(359 + k * .1, 360) * rad,
             !(k > 50 && k < 100));
    check(std::abs(f.yaw() - (359 + k * .1) * rad) < 1e-10);
    const auto &p = f.covariance();
    check(p[0] >= 0 && p[3] >= 0 && p[0] * p[3] - p[1] * p[2] >= -1e-18);
  }
  YawKalman bias;
  bias.initialize(0);
  for (int k = 0; k < 3000; ++k)
    bias.update(.4 * rad, .02, 0);
  check(std::abs(bias.bias() / rad - .4) < .01);
  const double old = bias.yaw();
  bool gap_rejected = false;
  try {
    bias.update(0, .3, 0);
  } catch (const std::runtime_error &) {
    gap_rejected = true;
  }
  check(gap_rejected);
  check(bias.yaw() == old);
  YawKalmanConfig c;
  c.heading_sigma_deg = 0;
  bool rejected = false;
  try {
    YawKalman invalid(c);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  check(rejected);
  std::cout
      << "Yaw KF bias convergence, wrap, missing heading and covariance PASS\n";
}
