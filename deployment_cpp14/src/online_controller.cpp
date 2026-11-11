#include "online_controller.hpp"
namespace deployment {
OnlineController::OnlineController(const Numbers &m, Mode mode, unsigned long long seed)
    : model_(m), actor_(m), critic_(m), mode_(mode), rng_(seed) {
  ensure(static_cast<int>(mode) >= 0 && static_cast<int>(mode) <= 3, "Unknown controller mode");
  horizon_ = integer(m, "horizon", 1, 10000);
  full_ = integer(m, "design", 0, 1) == 0;
  gamma_ = scalar(m, "gamma");
}
ControllerProposal OnlineController::propose(const std::vector<double> &obs) {
  // Keep RNG state unchanged if input validation/proposal fails.
  auto next_rng = rng_;
  std::normal_distribution<double> normal;
  std::vector<double> eps(8, 0);
  if (adapt_actor())
    for (double &x : eps)
      x = normal(next_rng);
  auto result = propose(obs, eps);
  rng_ = next_rng;
  return result;
}
ControllerProposal OnlineController::propose(const std::vector<double> &obs,
                                             const std::vector<double> &eps) {
  ensure(!waiting_ && !finished_, "Controller requires feedback or has finished");
  auto proposal = actor_.propose(obs, adapt_actor() ? eps : std::vector<double>(8, 0));
  ControllerProposal out;
  out.actor = proposal;
  out.q = mode_ == Mode::Fixed ? model_.at("fixed_q") : section(proposal.weights, 0, 6);
  out.r = mode_ == Mode::Fixed ? model_.at("fixed_r") : section(proposal.weights, 6, 2);
  out.p = mode_ == Mode::Fixed ? model_.at("fixed_qf") : critic_.p();
  obs_ = obs;
  latent_ = proposal.latent;
  waiting_ = true;
  return out;
}
void OnlineController::cancel() {
  ensure(waiting_, "No pending proposal");
  waiting_ = false;
  obs_.clear();
  latent_.clear();
}
OnlineUpdate OnlineController::observe(const std::vector<double> &error,
                                       const std::vector<double> &next_error, double reward,
                                       bool terminal, bool truncated) {
  // Whole transition is transactional, including delayed batch buffers/counters.
  auto candidate = *this;
  auto result = candidate.observe_impl(error, next_error, reward, terminal, truncated);
  *this = std::move(candidate);
  return result;
}
OnlineUpdate OnlineController::observe_impl(const std::vector<double> &e,
                                            const std::vector<double> &en, double reward,
                                            bool terminal, bool truncated) {
  ensure(waiting_ && !finished_, "No pending action to learn from");
  // Validate even fixed-mode feedback; Critic evaluates a candidate before mutation.
  auto step = critic_.evaluate(e, en, reward, terminal);
  OnlineUpdate out;
  if (mode_ != Mode::Fixed) {
    out.td = step.td;
    if (adapt_actor()) {
      if (horizon_ == 1) {
        out.actor_step_norm = actor_.apply(actor_.gradients(obs_, latent_, {step.td}));
        out.batch_size = 1;
        ++actor_updates_;
        ++actor_samples_;
      } else {
        if (rewards_.empty())
          snapshot_ = critic_.p();
        extend(pending_obs_, obs_);
        extend(pending_latent_, latent_);
        extend(errors_, e);
        rewards_.push_back(reward);
        auto collect = static_cast<std::size_t>(full_ ? 2 * horizon_ - 1 : horizon_);
        if (rewards_.size() >= collect || terminal || truncated) {
          out.returns =
              advantages(rewards_, errors_, en, snapshot_, gamma_, terminal, horizon_, full_);
          std::size_t count = out.returns.advantages.size();
          if (count) {
            out.actor_step_norm = actor_.apply(
                actor_.gradients(section(pending_obs_, 0, count * 17),
                                 section(pending_latent_, 0, count * 8), out.returns.advantages));
            out.batch_size = count;
            ++actor_updates_;
            actor_samples_ += count;
          }
          pending_obs_.clear();
          pending_latent_.clear();
          errors_.clear();
          rewards_.clear();
          snapshot_.clear();
        }
      }
    }
    if (adapt_critic()) {
      double norm = 0;
      for (int j = 0; j < 6; ++j) {
        double d = step.after[j] - critic_.p()[j];
        norm += d * d;
      }
      out.critic_step_norm = std::sqrt(norm);
      critic_.apply(step);
      ++critic_updates_;
    }
  }
  waiting_ = false;
  finished_ = terminal || truncated;
  return out;
}
Numbers OnlineController::model() const {
  auto out = model_;
  for (const auto &p : actor_.tensors())
    out[p.first] = p.second;
  out["critic_p"] = critic_.p();
  return out;
}
} // namespace deployment
