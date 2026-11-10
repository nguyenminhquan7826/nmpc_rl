#pragma once
#include <casadi/casadi.hpp>
#include <map>
#include <string>
#include <vector>
namespace deployment {
using Values = std::map<std::string, std::vector<double>>;
// Offline native solver. No serial connection or actuator side effects.
class FixedNmpc {
 public:
  explicit FixedNmpc(const Values& input);
  Values evaluate_and_solve();
  const std::string& status() const { return status_; }
 private:
  Values data_;
  std::string status_;
  int n_;
  casadi::Function dynamics_, evaluation_, solver_;
  std::vector<double> parameters_, lower_, upper_, glower_, gupper_;
};
}
