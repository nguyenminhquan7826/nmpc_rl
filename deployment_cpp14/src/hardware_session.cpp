#include "hardware_session.hpp"
#include <algorithm>
#include <chrono>
namespace deployment {
std::vector<double> HardwareSession::state(const Pose &p) const {
  std::vector<double> x{p.vx, 0, p.yaw, p.r, p.x, p.y};
  for (double v : x)
    ensure(std::isfinite(v), "Invalid real state");
  return x;
}
HardwareSession::HardwareSession(const Numbers &m, const Numbers &c,
                                 const ReferencePath &path, Mode mode,
                                 unsigned seed)
    : config_(c), path_(path), tracker_(path_), controller_(m, mode, seed) {
  n_ = integer(c, "N", 1, 200);
  ts_ = scalar(c, "Ts");
  ensure(ts_ >= .04 && ts_ <= .1, "Hardware Ts must be 40..100ms");
  limits_.max_speed = scalar(c, "max_speed");
  limits_.max_steering = scalar(c, "max_steering");
  limits_.servo_sign = integer(c, "servo_sign", -1, 1);
  ensure(limits_.servo_sign != 0, "Servo sign required");
  const auto p = field(c, "model", 9);
  drag_ = p[6] * p[7];
  ensure(std::abs(p[4] + p[5] - .25) < 1e-6,
         "Hardware wheelbase must be 0.25m");
  const auto lo = field(c, "input_lower", 2), hi = field(c, "input_upper", 2);
  ensure(lo[1] >= -limits_.max_steering && hi[1] <= limits_.max_steering,
         "Solver steering exceeds hardware limit");
  ensure(config_.at("state_lower").size() == 6 &&
             config_.at("state_upper").size() == 6,
         "Invalid state bound shape");
  ensure(config_.at("state_lower")[0] == 0 &&
             config_.at("state_upper")[0] <= limits_.max_speed,
         "Hardware requires forward-only vx bounds including standstill");
  hardware_command(0, drag_, 0, ts_, drag_, limits_);
  field(c, "task_q", 6);
  field(c, "task_r", 2);
  ensure(scalar(c, "feature_length") > 0, "Invalid feature length");
  for (const auto &k : {"task_q", "task_r"})
    for (double v : c.at(k))
      ensure(v > 0, "Invalid task weights");
  path_.validate_cad(scalar(c, "min_reference_speed"), limits_.max_speed, .25,
                     limits_.max_steering);
  end_ = path_.end();
  path_.append_tail(.2);
  auto x = path_.rows().front().state;
  std::vector<double> initial(x.begin(), x.end());
  previous_ = {drag_, 0};
  auto ref = tracker_.get(initial[4], initial[5], ts_, n_);
  config_["q"] = m.at("fixed_q");
  config_["r"] = m.at("fixed_r");
  config_["qf"] = m.at("fixed_qf");
  config_["state"] = initial;
  config_["previous"] = previous_;
  config_["reference"] = ref.states;
  config_["guess"] = std::vector<double>(8 * n_, 0);
  solver_.reset(new FixedNmpc(config_, true));
  tracker_.reset();
}
HardwareStep HardwareSession::propose(const Pose &p) {
  ensure(!proposed_, "Previous real proposal unresolved");
  auto x = state(p);
  using Clock = std::chrono::steady_clock;
  auto begin_reference = Clock::now();
  auto ref = tracker_.get(p.x, p.y, ts_, n_);
  auto xr = section(ref.states, 0, 6);
  auto begin_features = Clock::now();
  std::vector<double> meta{ref.projection.s,
                           path_.start(),
                           end_,
                           ref.curvatures.front(),
                           ref.curvatures.back(),
                           path_.rows().front().state[4],
                           path_.rows().front().state[5],
                           path_.rows().front().state[2],
                           scalar(config_, "feature_length")};
  auto obs = actor_features(x, xr, previous_, meta);
  auto begin_actor = Clock::now();
  auto proposal = controller_.propose(obs);
  auto end_actor = Clock::now();
  proposed_ = true;
  accepted_ = false;
  step_ = {};
  auto ms = [](Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  step_.reference_ms = ms(begin_reference, begin_features);
  step_.feature_ms = ms(begin_features, begin_actor);
  step_.actor_proposal_ms = ms(begin_actor, end_actor);
  step_.state = x;
  step_.reference = xr;
  step_.error = state_error(x, xr);
  step_.observation = obs;
  step_.mu = proposal.actor.mu;
  step_.sigma = proposal.actor.sigma;
  step_.latent = proposal.actor.latent;
  step_.q = proposal.q;
  step_.r = proposal.r;
  step_.p = proposal.p;
  step_.progress = ref.projection.s;
  step_.cte = ref.projection.distance;
  step_.reached = ref.projection.s >= end_ - .03;
  if (step_.reached)
    return step_;
  auto begin_warm_start = Clock::now();
  if (guess_.empty()) {
    auto predicted = x;
    std::vector<double> controls;
    for (int i = 0; i < n_; i++) {
      predicted = solver_->predict(predicted, previous_);
      extend(guess_, predicted);
      extend(controls, previous_);
    }
    extend(guess_, controls);
  }
  auto begin = Clock::now();
  step_.warm_start_ms = ms(begin_warm_start, begin);
  solver_->set_step(x, previous_, ref.states, guess_, proposal.q, proposal.r,
                    proposal.p);
  auto result = solver_->evaluate_and_solve();
  step_.solve_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - begin)
                       .count();
  step_.success = result.at("success")[0] == 1;
  if (step_.success) {
    step_.control = result.at("control");
    guess_ = result.at("shifted");
    auto begin_mapping = Clock::now();
    step_.command = hardware_command(
        p.vx, step_.control[0], step_.control[1], ts_, drag_, limits_,
        std::max(config_.at("input_lower")[1],
                 previous_[1] + config_.at("du_lower")[1]),
        std::min(config_.at("input_upper")[1],
                 previous_[1] + config_.at("du_upper")[1]));
    step_.applied = {(step_.command.speed - p.vx) / ts_ + drag_,
                     step_.command.steering};
    for (int j = 0; j < 2; j++) {
      double value = step_.applied[j], change = value - previous_[j];
      if (value < config_.at("input_lower")[j] - 1e-5 ||
          value > config_.at("input_upper")[j] + 1e-5 ||
          change < config_.at("du_lower")[j] - 1e-5 ||
          change > config_.at("du_upper")[j] + 1e-5)
        step_.command.clipped = true;
    }
    step_.command_mapping_ms = ms(begin_mapping, Clock::now());
  }
  return step_;
}
void HardwareSession::accept() {
  ensure(proposed_ && !accepted_ && step_.success && !step_.command.clipped,
         "Cannot accept failed/clipped/unproposed command");
  accepted_ = true;
}
OnlineUpdate HardwareSession::feedback(const Pose &p, bool terminal,
                                       bool truncated) {
  ensure(proposed_ && accepted_, "No accepted command for real feedback");
  auto x = state(p);
  auto ref = tracker_.get(p.x, p.y, ts_, n_);
  auto next_error = state_error(x, section(ref.states, 0, 6));
  double reward = 0;
  for (int i = 0; i < 6; i++)
    reward -= config_.at("task_q")[i] * step_.error[i] * step_.error[i];
  for (int i = 0; i < 2; i++) {
    double d = step_.applied[i] - previous_[i];
    reward -= config_.at("task_r")[i] * d * d;
  }
  reached_ = ref.projection.s >= end_ - .03;
  terminal = terminal || reached_;
  reward_ = reward;
  next_error_ = next_error;
  auto update =
      controller_.observe(step_.error, next_error, reward, terminal, truncated);
  previous_ = step_.applied;
  proposed_ = accepted_ = false;
  return update;
}
void HardwareSession::cancel() {
  if (proposed_)
    controller_.cancel();
  proposed_ = accepted_ = false;
}
} // namespace deployment
