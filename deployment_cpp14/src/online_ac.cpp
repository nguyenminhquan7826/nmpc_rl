#include "online_ac.hpp"
#include <algorithm>
#include <numeric>
namespace deployment {
namespace {
constexpr double pi = 3.14159265358979323846;
const std::map<std::string, std::size_t> shapes{
    {"W1", 17 * 64},  {"b1", 64},  {"W2", 64 * 64},     {"b2", 64},
    {"W_mu", 64 * 8}, {"b_mu", 8}, {"W_sigma", 64 * 8}, {"b_sigma", 8}};
void finite(const std::vector<double> &v, std::size_t n, const char *label) {
  ensure(v.size() == n, std::string("Wrong shape: ") + label);
  for (double x : v)
    ensure(std::isfinite(x), std::string("Nonfinite: ") + label);
}
void positive(const std::vector<double> &v) {
  for (double x : v)
    ensure(std::isfinite(x) && x > 0, "Require finite positive values");
}
void bounds(const std::vector<double> &lo, const std::vector<double> &hi) {
  positive(lo);
  for (std::size_t j = 0; j < lo.size(); ++j)
    ensure(hi[j] > lo[j], "Invalid bounds");
}
std::vector<double> affine(const std::vector<double> &x, const std::vector<double> &w,
                           const std::vector<double> &b) {
  auto y = b;
  for (std::size_t i = 0; i < x.size(); ++i)
    for (std::size_t j = 0; j < y.size(); ++j)
      y[j] += x[i] * w[i * y.size() + j];
  return y;
}
double value(const std::vector<double> &p, const std::vector<double> &e) {
  double v = 0;
  for (std::size_t j = 0; j < 6; ++j)
    v -= p[j] * e[j] * e[j];
  return v;
}
void outer_add(std::vector<double> &w, const std::vector<double> &x, const std::vector<double> &y) {
  for (std::size_t i = 0; i < x.size(); ++i)
    for (std::size_t j = 0; j < y.size(); ++j)
      w[i * y.size() + j] += x[i] * y[j];
}
} // namespace
void validate_model(const Numbers &m) {
  for (const auto &s : shapes)
    field(m, s.first, s.second);
  auto lo = field(m, "lower", 8), hi = field(m, "upper", 8);
  bounds(lo, hi);
  auto sb = field(m, "sigma_bounds", 2);
  ensure(sb[0] > 0 && sb[1] > sb[0], "Invalid sigma bounds");
  auto pl = field(m, "critic_lower", 6), pu = field(m, "critic_upper", 6),
       p = field(m, "critic_p", 6);
  bounds(pl, pu);
  for (int i = 0; i < 6; ++i)
    ensure(p[i] >= pl[i] && p[i] <= pu[i], "P outside bounds");
  auto q = field(m, "fixed_q", 6), r = field(m, "fixed_r", 2), qf = field(m, "fixed_qf", 6);
  positive(q);
  positive(r);
  positive(qf);
  for (int i = 0; i < 8; ++i) {
    double v = i < 6 ? q[i] : r[i - 6];
    ensure(v > lo[i] && v < hi[i], "Nominal Q/R must be interior");
  }
  for (int i = 0; i < 6; ++i)
    ensure(qf[i] >= pl[i] && qf[i] <= pu[i], "Fixed Qf outside P bounds");
  double gamma = scalar(m, "gamma");
  ensure(gamma > 0 && gamma <= 1, "Invalid gamma");
  ensure(scalar(m, "actor_lr") > 0, "Invalid Actor LR");
  positive(field(m, "critic_rates", 6));
  positive(field(m, "actor_factors", 8));
  auto reg = field(m, "mu_reg", 2);
  ensure(reg[0] >= 0 && reg[1] > 0, "Invalid mu regularization");
  integer(m, "critic_mode", 0, 1);
  integer(m, "horizon", 1, 10000);
  integer(m, "design", 0, 1);
  ensure(scalar(m, "train_length") > 0, "Invalid train length");
}
Actor::Actor(const Numbers &m) {
  validate_model(m);
  for (const auto &s : shapes)
    parameters_[s.first] = m.at(s.first);
  settings_ = {m.at("lower"),           m.at("upper"),           m.at("actor_factors"),
               m.at("sigma_bounds")[0], m.at("sigma_bounds")[1], scalar(m, "actor_lr"),
               m.at("mu_reg")[0],       m.at("mu_reg")[1]};
}
Actor::Forward Actor::forward(const std::vector<double> &obs) const {
  finite(obs, 17, "observation");
  Forward f;
  f.h1 = affine(obs, parameters_.at("W1"), parameters_.at("b1"));
  for (double &x : f.h1)
    x = std::tanh(x);
  f.h2 = affine(f.h1, parameters_.at("W2"), parameters_.at("b2"));
  for (double &x : f.h2)
    x = std::tanh(x);
  f.mu = affine(f.h2, parameters_.at("W_mu"), parameters_.at("b_mu"));
  f.fraction = affine(f.h2, parameters_.at("W_sigma"), parameters_.at("b_sigma"));
  for (double &x : f.fraction)
    x = .5 * (1 + std::tanh(.5 * x));
  f.sigma = f.fraction;
  for (double &x : f.sigma)
    x = settings_.sigma_lower + (settings_.sigma_upper - settings_.sigma_lower) * x;
  finite(f.mu, 8, "mu");
  finite(f.sigma, 8, "sigma");
  return f;
}
Proposal Actor::propose(const std::vector<double> &observation,
                        const std::vector<double> &epsilon) const {
  finite(epsilon, 8, "epsilon");
  auto f = forward(observation);
  Proposal out{f.mu, f.sigma, {}, {}, 0};
  for (int j = 0; j < 8; ++j) {
    double z = f.mu[j] + f.sigma[j] * epsilon[j];
    out.latent.push_back(z);
    out.weights.push_back(settings_.lower[j] +
                          .5 * (settings_.upper[j] - settings_.lower[j]) * (1 + std::tanh(z)));
    double d = (z - f.mu[j]) / f.sigma[j];
    out.log_probability -= .5 * (d * d + 2 * std::log(f.sigma[j]) + std::log(2 * pi));
  }
  finite(out.latent, 8, "sampled latent");
  ensure(std::isfinite(out.log_probability), "Nonfinite log probability");
  return out;
}
ActorGradient Actor::gradients(const std::vector<double> &obs, const std::vector<double> &zs,
                               const std::vector<double> &adv) const {
  std::size_t n = adv.size();
  ensure(n > 0, "Empty Actor batch");
  finite(obs, n * 17, "batch observations");
  finite(zs, n * 8, "latents");
  finite(adv, n, "advantages");
  ActorGradient out{0, {}};
  for (const auto &s : shapes)
    out.tensors[s.first] = std::vector<double>(s.second, 0);
  for (std::size_t b = 0; b < n; ++b) {
    auto x = section(obs, b * 17, 17), z = section(zs, b * 8, 8);
    auto f = forward(x);
    std::vector<double> dmu(8), draw(8), dh2(64, 0), dh1(64, 0);
    double lp = 0, penalty = 0;
    for (int j = 0; j < 8; ++j) {
      double d = z[j] - f.mu[j], inv = 1 / f.sigma[j], factor = settings_.factors[j];
      lp -= .5 * (d * d * inv * inv + 2 * std::log(f.sigma[j]) + std::log(2 * pi));
      double excess = std::max(std::abs(f.mu[j]) - settings_.mu_threshold, 0.0);
      penalty += excess * excess;
      dmu[j] = -adv[b] / n * d * inv * inv * factor +
               2 * settings_.mu_penalty / n * (f.mu[j] > 0 ? 1 : (f.mu[j] < 0 ? -1 : 0)) * excess;
      draw[j] = -adv[b] / n * (d * d * inv * inv - 1) * inv *
                (settings_.sigma_upper - settings_.sigma_lower) * f.fraction[j] *
                (1 - f.fraction[j]) * factor;
      out.tensors["b_mu"][j] += dmu[j];
      out.tensors["b_sigma"][j] += draw[j];
    }
    out.loss += (-adv[b] * lp + settings_.mu_penalty * penalty) / n;
    outer_add(out.tensors["W_mu"], f.h2, dmu);
    outer_add(out.tensors["W_sigma"], f.h2, draw);
    for (int i = 0; i < 64; ++i) {
      for (int j = 0; j < 8; ++j)
        dh2[i] += dmu[j] * parameters_.at("W_mu")[i * 8 + j] +
                  draw[j] * parameters_.at("W_sigma")[i * 8 + j];
      dh2[i] *= 1 - f.h2[i] * f.h2[i];
      out.tensors["b2"][i] += dh2[i];
    }
    outer_add(out.tensors["W2"], f.h1, dh2);
    for (int i = 0; i < 64; ++i) {
      for (int j = 0; j < 64; ++j)
        dh1[i] += dh2[j] * parameters_.at("W2")[i * 64 + j];
      dh1[i] *= 1 - f.h1[i] * f.h1[i];
      out.tensors["b1"][i] += dh1[i];
    }
    outer_add(out.tensors["W1"], x, dh1);
  }
  ensure(std::isfinite(out.loss), "Nonfinite Actor loss");
  for (const auto &s : shapes)
    finite(out.tensors.at(s.first), s.second, "gradient");
  return out;
}
double Actor::apply(const ActorGradient &g) {
  auto candidate = parameters_;
  double norm2 = 0;
  for (const auto &s : shapes) {
    const auto &grad = field(g.tensors, s.first, s.second);
    auto &dst = candidate.at(s.first);
    for (std::size_t i = 0; i < grad.size(); ++i) {
      double step = settings_.learning_rate * grad[i];
      dst[i] -= step;
      norm2 += step * step;
    }
    finite(dst, s.second, "Actor candidate");
  }
  ensure(std::isfinite(norm2), "Nonfinite Actor step norm");
  parameters_.swap(candidate);
  return std::sqrt(norm2);
}
Critic::Critic(const Numbers &m) {
  validate_model(m);
  p_ = m.at("critic_p");
  lower_ = m.at("critic_lower");
  upper_ = m.at("critic_upper");
  rates_ = m.at("critic_rates");
  gamma_ = scalar(m, "gamma");
  residual_ = integer(m, "critic_mode", 0, 1) == 1;
}
CriticUpdate Critic::evaluate(const std::vector<double> &e, const std::vector<double> &en,
                              double reward, bool terminal) const {
  finite(e, 6, "error");
  finite(en, 6, "next error");
  ensure(std::isfinite(reward), "Nonfinite reward");
  double v = value(p_, e), vn = terminal ? 0 : value(p_, en), td = reward + gamma_ * vn - v;
  CriticUpdate out{td, .5 * td * td, v, vn, {}, {}};
  ensure(std::isfinite(out.loss), "Nonfinite TD loss");
  for (int j = 0; j < 6; ++j) {
    double grad = td * (e[j] * e[j] - (residual_ && !terminal ? gamma_ * en[j] * en[j] : 0));
    double next = p_[j] - rates_[j] * grad;
    ensure(std::isfinite(grad) && std::isfinite(next), "Nonfinite Critic candidate");
    out.gradient.push_back(grad);
    out.after.push_back(std::max(lower_[j], std::min(upper_[j], next)));
  }
  return out;
}
void Critic::apply(const CriticUpdate &update) {
  finite(update.after, 6, "Critic candidate");
  for (int j = 0; j < 6; ++j)
    ensure(update.after[j] >= lower_[j] && update.after[j] <= upper_[j],
           "Critic candidate outside bounds");
  p_ = update.after;
}
std::vector<double> state_error(const std::vector<double> &x, const std::vector<double> &xr) {
  finite(x, 6, "state");
  finite(xr, 6, "reference");
  std::vector<double> e(6);
  for (int j = 0; j < 6; ++j)
    e[j] = x[j] - xr[j];
  e[2] = std::atan2(std::sin(e[2]), std::cos(e[2]));
  finite(e, 6, "state error");
  return e;
}
std::vector<double> actor_features(const std::vector<double> &x, const std::vector<double> &xr,
                                   const std::vector<double> &u, const std::vector<double> &m) {
  auto e = state_error(x, xr);
  finite(u, 2, "previous control");
  finite(m, 9, "feature metadata");
  ensure(m[8] > 0 && m[2] > m[1], "Invalid path length");
  auto physical = x;
  physical[2] -= m[7];
  physical[4] -= m[5];
  physical[5] -= m[6];
  double scale[6] = {.1, .06, pi, .5, m[8], m[8]}, es[6] = {.1, .06, .2, .5, .1, .1};
  std::vector<double> out;
  for (int j = 0; j < 6; ++j)
    out.push_back(physical[j] / scale[j]);
  for (int j = 0; j < 6; ++j)
    out.push_back(e[j] / es[j]);
  out.push_back(u[0] / 2);
  out.push_back(u[1] / .55);
  out.push_back((m[0] - m[1]) / (m[2] - m[1]));
  out.push_back(m[3] / 2);
  out.push_back(m[4] / 2);
  finite(out, 17, "features");
  return out;
}
Returns advantages(const std::vector<double> &rewards, const std::vector<double> &errors,
                   const std::vector<double> &en, const std::vector<double> &p, double gamma,
                   bool terminal, int horizon, bool full) {
  std::size_t n = rewards.size();
  ensure(n > 0, "Empty rollout");
  finite(rewards, n, "rewards");
  finite(errors, n * 6, "errors");
  finite(en, 6, "next error");
  finite(p, 6, "P snapshot");
  ensure(std::isfinite(gamma) && gamma > 0 && gamma <= 1 && horizon > 0, "Invalid return settings");
  Returns out;
  if (!full) {
    out.targets.resize(n);
    double tail = terminal ? 0 : value(p, en);
    for (std::size_t j = n; j-- > 0;) {
      tail = rewards[j] + gamma * tail;
      out.targets[j] = tail;
      out.advantages.insert(out.advantages.begin(), tail - value(p, section(errors, j * 6, 6)));
      out.horizons.insert(out.horizons.begin(), static_cast<double>(n - j));
    }
  } else {
    std::size_t count =
        terminal ? n : (n >= static_cast<std::size_t>(horizon) ? n - horizon + 1 : 0);
    for (std::size_t i = 0; i < count; ++i) {
      std::size_t end = std::min(i + static_cast<std::size_t>(horizon), n);
      double target = 0, discount = 1;
      for (std::size_t j = i; j < end; ++j) {
        target += discount * rewards[j];
        discount *= gamma;
      }
      if (!(terminal && end == n))
        target += discount * value(p, end < n ? section(errors, end * 6, 6) : en);
      out.targets.push_back(target);
      out.advantages.push_back(target - value(p, section(errors, i * 6, 6)));
      out.horizons.push_back(static_cast<double>(end - i));
    }
  }
  for (double x : out.advantages)
    ensure(std::isfinite(x), "Nonfinite advantage");
  return out;
}
} // namespace deployment
