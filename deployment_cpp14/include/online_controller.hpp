#pragma once
#include "online_ac.hpp"
namespace deployment {
enum class Mode { Fixed = 0, ActorAdapt = 1, CriticAdapt = 2, FullAdapt = 3 };
struct ControllerProposal {
  Proposal actor;
  std::vector<double> q, r, p;
};
struct OnlineUpdate {
  double td = 0, actor_step_norm = 0, critic_step_norm = 0;
  std::size_t batch_size = 0;
  Returns returns;
};
class OnlineController {
public:
  OnlineController(const Numbers &model, Mode mode, unsigned long long seed);
  ControllerProposal propose(const std::vector<double> &observation);
  ControllerProposal propose(const std::vector<double> &observation,
                             const std::vector<double> &epsilon);
  OnlineUpdate observe(const std::vector<double> &error, const std::vector<double> &next_error,
                       double reward, bool terminal, bool truncated);
  // On solver failure discard this proposal. No learning from unapplied action.
  void cancel();
  Numbers model() const;
  std::size_t actor_updates() const { return actor_updates_; }
  std::size_t actor_samples() const { return actor_samples_; }
  std::size_t critic_updates() const { return critic_updates_; }

private:
  OnlineUpdate observe_impl(const std::vector<double> &error, const std::vector<double> &next_error,
                            double reward, bool terminal, bool truncated);
  Numbers model_;
  Actor actor_;
  Critic critic_;
  Mode mode_;
  std::mt19937_64 rng_;
  bool waiting_ = false, finished_ = false;
  std::vector<double> obs_, latent_, pending_obs_, pending_latent_, errors_, rewards_, snapshot_;
  int horizon_;
  bool full_;
  double gamma_;
  std::size_t actor_updates_ = 0, actor_samples_ = 0, critic_updates_ = 0;
  bool adapt_actor() const { return mode_ == Mode::ActorAdapt || mode_ == Mode::FullAdapt; }
  bool adapt_critic() const { return mode_ == Mode::CriticAdapt || mode_ == Mode::FullAdapt; }
};
} // namespace deployment
