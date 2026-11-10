#include "fixed_nmpc.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace deployment {
using casadi::DM;
using casadi::Slice;
using casadi::SX;
namespace {
void append(std::vector<double> &a, const std::vector<double> &b) {
  a.insert(a.end(), b.begin(), b.end());
}
std::vector<double> dense(const DM &a) { return DM::densify(a).nonzeros(); }
void require(bool ok, const std::string &message) {
  if (!ok)
    throw std::invalid_argument(message);
}
} // namespace
FixedNmpc::FixedNmpc(const Values &input, bool dynamic_weights)
    : data_(input), n_(0), dynamic_(dynamic_weights) {
  auto field = [&](const std::string &key, int size,
                   bool bounds = false) -> const std::vector<double> & {
    auto it = data_.find(key);
    require(it != data_.end(), "Missing " + key);
    require(static_cast<int>(it->second.size()) == size, "Wrong size: " + key);
    for (double v : it->second)
      require(bounds ? !std::isnan(v) : std::isfinite(v), "Nonfinite: " + key);
    return it->second;
  };
  double horizon = field("N", 1)[0];
  require(horizon >= 1 && horizon <= 1000 && std::floor(horizon) == horizon, "Invalid horizon");
  n_ = static_cast<int>(horizon);
  double ts = field("Ts", 1)[0];
  require(ts > 0, "Invalid Ts");
  double iterations = field("sqp_max_iter", 1)[0];
  require(iterations >= 1 && iterations <= 10000 && std::floor(iterations) == iterations,
          "Invalid SQP iterations");
  for (const auto &k : {"q", "r", "qf"})
    for (double v : field(k, std::string(k) == "r" ? 2 : 6))
      require(v > 0, "Nonpositive weight");
  auto model = field("model", 9);
  for (int i = 0; i < 9; ++i)
    require(model[i] > 0, "Nonpositive model parameter");
  for (const auto &kind : {"state", "input", "du"}) {
    int size = std::string(kind) == "state" ? 6 : 2;
    auto lo = field(std::string(kind) + "_lower", size, true),
         hi = field(std::string(kind) + "_upper", size, true);
    for (int i = 0; i < size; ++i)
      require(lo[i] < hi[i] && lo[i] != std::numeric_limits<double>::infinity() &&
                  hi[i] != -std::numeric_limits<double>::infinity(),
              "Invalid bounds");
  }
  require(field("beta_max", 1)[0] > 0 && field("vy_abs_max", 1)[0] > 0, "Invalid lateral limits");
  field("state", 6);
  field("previous", 2);
  field("reference", 6 * (n_ + 1));
  field("guess", 8 * n_);
  for (const auto &key : {"state", "previous", "reference", "model"})
    append(parameters_, data_.at(key));
  const int weight_offset = static_cast<int>(parameters_.size());
  if (dynamic_)
    for (const auto &key : {"q", "r", "qf"})
      append(parameters_, data_.at(key));
  SX x = SX::sym("x", 6), u = SX::sym("u", 2), p = SX::sym("model", 9);
  SX vx = x(0), vy = x(1), psi = x(2), rate = x(3), delta = u(1);
  SX safe = sqrt(vx * vx + p(8) * p(8));
  SX fyf = p(2) * (delta - (vy + p(4) * rate) / safe), fyr = -p(3) * (vy - p(5) * rate) / safe;
  SX rhs = SX::vertcat({u(0) - fyf / p(0) * sin(delta) - p(6) * p(7) + vy * rate,
                        (fyr + fyf * cos(delta)) / p(0) - vx * rate, rate,
                        (p(4) * fyf * cos(delta) - p(5) * fyr) / p(1),
                        vx * cos(psi) - vy * sin(psi), vx * sin(psi) + vy * cos(psi)});
  casadi::Function continuous("continuous", {x, u, p}, {rhs});
  auto f = [&](const SX &state) { return continuous(std::vector<SX>{state, u, p}).at(0); };
  SX k1 = f(x), k2 = f(x + ts / 2 * k1), k3 = f(x + ts / 2 * k2), k4 = f(x + ts * k3);
  dynamics_ = casadi::Function("rk4", {x, u, p}, {x + ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)});
  SX X = SX::sym("X", 6, n_), U = SX::sym("U", 2, n_),
     P = SX::sym("P", static_cast<int>(parameters_.size()));
  SX previous = P(Slice(6, 8)), refs = SX::reshape(P(Slice(8, 8 + 6 * (n_ + 1))), 6, n_ + 1),
     pm = P(Slice(8 + 6 * (n_ + 1), weight_offset));
  SX xp = P(Slice(0, 6)), cost = 0;
  std::vector<SX> constraints;
  const double inf = std::numeric_limits<double>::infinity();
  for (int i = 0; i < n_; ++i) {
    SX xi = X(Slice(), i), ui = U(Slice(), i), du = ui - previous;
    constraints.push_back(xi - dynamics_(std::vector<SX>{xp, ui, pm}).at(0));
    append(glower_, std::vector<double>(6, 0));
    append(gupper_, std::vector<double>(6, 0));
    constraints.push_back(du);
    append(glower_, data_.at("du_lower"));
    append(gupper_, data_.at("du_upper"));
    double beta = data_.at("beta_max")[0];
    constraints.push_back(
        SX::vertcat({xi(1) - beta * xi(0), -xi(1) - beta * xi(0), xi(1), -xi(1)}));
    append(glower_, {-inf, -inf, -inf, -inf});
    append(gupper_, {0, 0, data_.at("vy_abs_max")[0], data_.at("vy_abs_max")[0]});
    SX e = xi - refs(Slice(), i + 1);
    e(2) = atan2(sin(e(2)), cos(e(2)));
    const auto &weights = data_.at(i == n_ - 1 ? "qf" : "q");
    // Match Python diagonal matrix multiplication accumulation order.
    SX state_cost = 0, control_cost = 0;
    for (int j = 0; j < 6; ++j) {
      SX w = dynamic_ ? P(weight_offset + (i == n_ - 1 ? 8 : 0) + j) : SX(weights[j]);
      state_cost += e(j) * w * e(j);
    }
    for (int j = 0; j < 2; ++j) {
      SX w = dynamic_ ? P(weight_offset + 6 + j) : SX(data_.at("r")[j]);
      control_cost += du(j) * w * du(j);
    }
    cost += state_cost;
    cost += control_cost;
    xp = xi;
    previous = ui;
    append(lower_, data_.at("state_lower"));
    append(upper_, data_.at("state_upper"));
  }
  for (int i = 0; i < n_; ++i) {
    append(lower_, data_.at("input_lower"));
    append(upper_, data_.at("input_upper"));
  }
  SX z = SX::vertcat({SX::reshape(X, 6 * n_, 1), SX::reshape(U, 2 * n_, 1)}),
     g = SX::vertcat(constraints);
  evaluation_ =
      casadi::Function("evaluation", {z, P}, {cost, g, SX::gradient(cost, z), SX::jacobian(g, z)});
  casadi::Dict qp{{"print_header", false}, {"print_iter", false}};
  casadi::Dict opts{{"qpsol", "qrqp"},       {"max_iter", static_cast<int>(iterations)},
                    {"print_header", false}, {"print_iteration", false},
                    {"print_status", false}, {"print_time", false},
                    {"qpsol_options", qp}};
  solver_ = casadi::nlpsol("fixed_nmpc", "sqpmethod",
                           casadi::SXDict{{"x", z}, {"p", P}, {"f", cost}, {"g", g}}, opts);
}
void FixedNmpc::set_step(const std::vector<double> &state, const std::vector<double> &previous,
                         const std::vector<double> &reference, const std::vector<double> &guess,
                         const std::vector<double> &q, const std::vector<double> &r,
                         const std::vector<double> &qf) {
  Values next{{"state", state},
              {"previous", previous},
              {"reference", reference},
              {"guess", guess},
              {"q", q},
              {"r", r},
              {"qf", qf}};
  for (const auto &item : next) {
    int expected = item.first == "guess"
                       ? 8 * n_
                       : (item.first == "reference"
                              ? 6 * (n_ + 1)
                              : ((item.first == "r" || item.first == "previous") ? 2 : 6));
    require(static_cast<int>(item.second.size()) == expected, "Invalid step shape: " + item.first);
    for (double v : item.second)
      require(std::isfinite(v), "Nonfinite step input");
  }
  for (const auto &key : {"q", "r", "qf"})
    for (double v : next.at(key))
      require(v > 0, "Nonpositive step weight");
  if (!dynamic_)
    for (const auto &key : {"q", "r", "qf"})
      require(next.at(key) == data_.at(key), "Fixed solver weights cannot change");
  std::vector<double> par;
  for (const auto &key : {"state", "previous", "reference"})
    append(par, next.at(key));
  append(par, data_.at("model"));
  if (dynamic_)
    for (const auto &key : {"q", "r", "qf"})
      append(par, next.at(key));
  for (const auto &item : next)
    data_[item.first] = item.second;
  parameters_.swap(par);
}
std::vector<double> FixedNmpc::predict(const std::vector<double> &state,
                                       const std::vector<double> &control) const {
  require(state.size() == 6 && control.size() == 2, "Invalid prediction shape");
  for (double v : state)
    require(std::isfinite(v), "Nonfinite prediction state");
  for (double v : control)
    require(std::isfinite(v), "Nonfinite prediction control");
  auto result =
      dense(dynamics_(std::vector<DM>{DM(state), DM(control), DM(data_.at("model"))}).at(0));
  for (double v : result)
    require(std::isfinite(v), "Nonfinite prediction result");
  return result;
}
Values FixedNmpc::evaluate_and_solve() {
  auto ev = evaluation_(std::vector<DM>{DM(data_.at("guess")), DM(parameters_)});
  Values out{
      {"parameters", parameters_},
      {"eval_f", dense(ev[0])},
      {"eval_g", dense(ev[1])},
      {"eval_gradient", dense(ev[2])},
      {"eval_jacobian", dense(ev[3])},
      {"rk4", dense(dynamics_(std::vector<DM>{DM(data_.at("state")), DM(data_.at("previous")),
                                              DM(data_.at("model"))})
                        .at(0))},
      {"success", {0}},
      {"solver_status_code", {4}},
      {"max_violation", {std::numeric_limits<double>::infinity()}},
      {"control", {}},
      {"shifted", {}}};
  try {
    auto sol = solver_(casadi::DMDict{{"x0", DM(data_.at("guess"))},
                                      {"p", DM(parameters_)},
                                      {"lbx", DM(lower_)},
                                      {"ubx", DM(upper_)},
                                      {"lbg", DM(glower_)},
                                      {"ubg", DM(gupper_)}});
    auto z = dense(sol.at("x")), g = dense(sol.at("g"));
    double violation = 0;
    auto check = [&](const std::vector<double> &v, const std::vector<double> &lo,
                     const std::vector<double> &hi) {
      for (std::size_t i = 0; i < v.size(); ++i) {
        if (!std::isfinite(v[i])) {
          violation = std::numeric_limits<double>::infinity();
          return;
        }
        violation = std::max(violation, std::max(lo[i] - v[i], v[i] - hi[i]));
      }
    };
    check(z, lower_, upper_);
    check(g, glower_, gupper_);
    out["max_violation"] = {violation};
    out["solution"] = z;
    out["objective"] = dense(sol.at("f"));
    auto stats = solver_.stats();
    status_ = stats.at("return_status").to_string();
    bool finite = std::isfinite(out["objective"][0]);
    bool solved = stats.at("success").to_bool();
    bool success = solved && violation <= 1e-6 && finite;
    out["solver_status_code"] = {success ? 0.0 : (!finite ? 3.0 : (!solved ? 1.0 : 2.0))};
    if (success) {
      out["success"] = {1};
      out["control"] = {z[6 * n_], z[6 * n_ + 1]};
      for (int i = 0; i < n_; ++i) {
        int src = std::min(i + 1, n_ - 1) * 6;
        append(out["shifted"], std::vector<double>(z.begin() + src, z.begin() + src + 6));
      }
      for (int i = 0; i < n_; ++i) {
        int src = 6 * n_ + std::min(i + 1, n_ - 1) * 2;
        append(out["shifted"], std::vector<double>(z.begin() + src, z.begin() + src + 2));
      }
    }
  } catch (const std::exception &error) {
    status_ = error.what(); /* No new control is valid. */
  }
  return out;
}
} // namespace deployment
