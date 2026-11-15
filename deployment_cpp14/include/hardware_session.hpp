#pragma once
#include "fixed_nmpc.hpp"
#include "hardware_control.hpp"
#include "online_controller.hpp"
#include "reference_path.hpp"
#include <memory>
namespace deployment {
struct HardwareStep {
  std::vector<double> state, reference, error, observation, q, r, p, control,
      applied, mu, sigma, latent;
  HardwareCommand command{};
  double progress = 0, cte = 0, solve_ms = 0;
  bool success = false, reached = false;
};
class HardwareSession {
  Numbers config_;
  ReferencePath path_;
  RouteTracker tracker_;
  OnlineController controller_;
  HardwareLimits limits_;
  std::unique_ptr<FixedNmpc> solver_;
  std::vector<double> previous_, guess_;
  HardwareStep step_;
  double reward_ = 0;
  std::vector<double> next_error_;
  bool proposed_ = false, accepted_ = false, reached_ = false;
  int n_;
  double ts_, end_, drag_;
  std::vector<double> state(const Pose &p) const;

public:
  HardwareSession(const Numbers &model, const Numbers &settings,
                  const ReferencePath &path, Mode mode, unsigned seed);
  HardwareStep propose(const Pose &p);
  void accept();
  OnlineUpdate feedback(const Pose &p, bool terminal, bool truncated);
  void cancel();
  Numbers model() const { return controller_.model(); }
  std::size_t actor_updates() const { return controller_.actor_updates(); }
  std::size_t critic_updates() const { return controller_.critic_updates(); }
  double period() const { return ts_; }
  bool reached() const { return reached_; }
  double reward() const { return reward_; }
  const std::vector<double> &next_error() const { return next_error_; }
};
} // namespace deployment
