#pragma once
#include "numeric_io.hpp"
#include <random>
namespace deployment {
struct ActorSettings {
  std::vector<double> lower, upper, factors;
  double sigma_lower, sigma_upper, learning_rate, mu_penalty, mu_threshold;
};
struct Proposal {
  std::vector<double> mu, sigma, latent, weights;
  double log_probability;
};
struct ActorGradient {
  double loss;
  Numbers tensors;
};
class Actor {
public:
  explicit Actor(const Numbers &model);
  Proposal propose(const std::vector<double> &observation,
                   const std::vector<double> &epsilon) const;
  ActorGradient gradients(const std::vector<double> &observations,
                          const std::vector<double> &latents,
                          const std::vector<double> &advantages) const;
  double apply(const ActorGradient &gradient);
  const Numbers &tensors() const { return parameters_; }

private:
  ActorSettings settings_;
  Numbers parameters_;
  struct Forward {
    std::vector<double> h1, h2, mu, sigma, fraction;
  };
  Forward forward(const std::vector<double> &observation) const;
};
struct CriticUpdate {
  double td, loss, value_current, value_next;
  std::vector<double> gradient, after;
};
class Critic {
public:
  explicit Critic(const Numbers &model);
  CriticUpdate evaluate(const std::vector<double> &error, const std::vector<double> &next_error,
                        double reward, bool terminal) const;
  void apply(const CriticUpdate &update);
  const std::vector<double> &p() const { return p_; }

private:
  std::vector<double> p_, lower_, upper_, rates_;
  double gamma_;
  bool residual_;
};
struct Returns {
  std::vector<double> advantages, targets, horizons;
};
Returns advantages(const std::vector<double> &rewards, const std::vector<double> &errors,
                   const std::vector<double> &next_error, const std::vector<double> &p,
                   double gamma, bool terminal, int horizon, bool full_horizon);
std::vector<double> state_error(const std::vector<double> &state,
                                const std::vector<double> &reference);
// meta: progress,start,end,curvature_now,curvature_end,originX,originY,heading0,train_length.
std::vector<double> actor_features(const std::vector<double> &state,
                                   const std::vector<double> &reference,
                                   const std::vector<double> &previous,
                                   const std::vector<double> &meta);
void validate_model(const Numbers &model);
} // namespace deployment
