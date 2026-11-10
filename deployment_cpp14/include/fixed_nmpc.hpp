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
  explicit FixedNmpc(const Values &input, bool dynamic_weights = false);
  void set_step(const std::vector<double> &state, const std::vector<double> &previous,
                const std::vector<double> &reference, const std::vector<double> &guess,
                const std::vector<double> &q, const std::vector<double> &r,
                const std::vector<double> &qf);
  std::vector<double> predict(const std::vector<double> &state,
                              const std::vector<double> &control) const;
  Values evaluate_and_solve();
  const std::string &status() const { return status_; }

private:
  Values data_;
  std::string status_;
  int n_;
  bool dynamic_;
  casadi::Function dynamics_, evaluation_, solver_;
  std::vector<double> parameters_, lower_, upper_, glower_, gupper_;
};
} // namespace deployment
