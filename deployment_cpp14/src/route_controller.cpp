#include "fixed_nmpc.hpp"
#include "online_controller.hpp"
#include "reference_path.hpp"
#include <algorithm>
#include <iostream>
#include <limits>
namespace {
using namespace deployment;
void write_csv(const std::string &file, const std::vector<Numbers> &rows) {
  ensure(!rows.empty(), "No route log rows");
  std::ofstream out(file);
  ensure(bool(out), "Cannot open route log");
  out << std::setprecision(17);
  bool first = true;
  for (const auto &item : rows[0])
    for (std::size_t j = 0; j < item.second.size(); ++j) {
      if (!first)
        out << ',';
      first = false;
      out << item.first;
      if (item.second.size() > 1)
        out << '_' << j;
    }
  out << '\n';
  for (const auto &row : rows) {
    first = true;
    for (const auto &item : rows[0]) {
      const auto &v = row.at(item.first);
      ensure(v.size() == item.second.size(), "Inconsistent log shape");
      for (double x : v) {
        if (!first)
          out << ',';
        first = false;
        out << x;
      }
    }
    out << '\n';
  }
  out.flush();
  ensure(bool(out), "Route log write failed");
}
std::vector<double> rollout_guess(const ReferencePath &path, const RouteReference &ref,
                                  const Values &config, const FixedNmpc &solver,
                                  const std::vector<double> &state,
                                  const std::vector<double> &previous, bool rate_limits) {
  int n = integer(config, "N", 1, 1000);
  const auto &p = config.at("model");
  auto x = state, prev = previous;
  std::vector<double> states, controls;
  for (int i = 0; i < n; ++i) {
    double delta = std::atan((p[4] + p[5]) * path.curvature(ref.progress[i]));
    std::vector<double> u{p[6] * p[7], std::max(config.at("input_lower")[1],
                                                std::min(config.at("input_upper")[1], delta))};
    if (rate_limits)
      for (int j = 0; j < 2; ++j) {
        double lo = std::max(config.at("input_lower")[j], prev[j] + config.at("du_lower")[j]),
               hi = std::min(config.at("input_upper")[j], prev[j] + config.at("du_upper")[j]);
        ensure(lo <= hi, "Incompatible absolute/rate bounds");
        u[j] = std::max(lo, std::min(hi, u[j]));
      }
    x = solver.predict(x, u);
    extend(states, x);
    extend(controls, u);
    prev = u;
  }
  extend(states, controls);
  return states;
}
} // namespace
int main(int argc, char **argv) {
  try {
    using namespace deployment;
    ensure(argc == 6, "usage: route_controller model route.csv settings output.csv mode");
    ensure(!std::ifstream(argv[4]) && !std::ifstream(std::string(argv[4]) + ".model") &&
               !std::ifstream(std::string(argv[4]) + ".summary"),
           "Use new route output paths");
    auto model = read_numbers(argv[1], "AC_MODEL_V1"),
         config = read_numbers(argv[3], "NMPC_ROUTE_V1");
    const std::map<std::string, Mode> modes{{"fixed", Mode::Fixed},
                                            {"actor_adapt", Mode::ActorAdapt},
                                            {"critic_adapt", Mode::CriticAdapt},
                                            {"full_adapt", Mode::FullAdapt}};
    ensure(modes.count(argv[5]), "Unknown mode");
    Mode mode = modes.at(argv[5]);
    int max_steps = integer(config, "max_steps", 1, 100000), n = integer(config, "N", 1, 1000);
    double ts = scalar(config, "Ts");
    int seed = integer(config, "seed", 0, 2147483647);
    OnlineController controller(model, mode, static_cast<unsigned long long>(seed));
    auto path = ReferencePath::load_csv(argv[2]);
    const auto p = field(config, "model", 9);
    double wheelbase = p[4] + p[5];
    double min_speed = scalar(config, "min_reference_speed"),
           max_speed = scalar(config, "max_reference_speed");
    field(config, "input_lower", 2);
    field(config, "input_upper", 2);
    path.validate_cad(
        min_speed, max_speed, wheelbase,
        std::min(std::abs(config.at("input_lower")[1]), std::abs(config.at("input_upper")[1])));
    double cad_end = path.end();
    path.append_tail(scalar(config, "tail_length"));
    double feature_length = scalar(config, "feature_length");
    ensure(feature_length > 0, "Invalid feature length");
    double solver_penalty = scalar(config, "solver_penalty"),
           failure_penalty = scalar(config, "failure_penalty");
    ensure(solver_penalty >= 0 && failure_penalty >= 0, "Invalid learning penalties");
    const auto qe = field(config, "task_q", 6);
    const auto re = field(config, "task_r", 2);
    for (double x : qe)
      ensure(x > 0, "Invalid task Qe");
    for (double x : re)
      ensure(x > 0, "Invalid task Re");
    bool supplied = config.count("epsilon");
    std::vector<double> epsilon;
    if (supplied)
      epsilon = field(config, "epsilon", max_steps * 8);
    RouteTracker tracker(path);
    auto ref0 = path.interpolate(path.start());
    double ey = scalar(config, "ey"), heading = scalar(config, "heading_rad");
    auto state = ref0;
    state[4] = ref0[4] - ey * std::sin(ref0[2]);
    state[5] = ref0[5] + ey * std::cos(ref0[2]);
    state[2] = std::atan2(std::sin(ref0[2] + heading), std::cos(ref0[2] + heading));
    state[1] = 0;
    state[3] = 0;
    tracker.project(state[4], state[5]);
    auto initial_ref = tracker.get(state[4], state[5], ts, n);
    std::vector<double> previous{p[6] * p[7], 0};
    config["q"] = model.at("fixed_q");
    config["r"] = model.at("fixed_r");
    config["qf"] = model.at("fixed_qf");
    config["state"] = state;
    config["previous"] = previous;
    config["reference"] = initial_ref.states;
    config["guess"] = std::vector<double>(8 * n, 0);
    FixedNmpc solver(config, true);
    auto guess = rollout_guess(path, initial_ref, config, solver, state, previous, false);
    auto ref = tracker.get(state[4], state[5], ts, n);
    std::vector<Numbers> rows;
    int fail_streak = 0;
    bool reached = false, catastrophic = false, terminated = false, truncated = false;
    double total_reward = 0, discounted_cost = 0, discount = 1, cte_sum = 0, cte_max = 0;
    std::size_t successes = 0, pending_start = 0, pending_count = 0;
    for (int k = 0; k < max_steps; ++k) {
      auto xr = section(ref.states, 0, 6);
      std::vector<double> meta{ref.projection.s,
                               path.start(),
                               cad_end,
                               ref.curvatures.front(),
                               ref.curvatures.back(),
                               path.rows()[0].state[4],
                               path.rows()[0].state[5],
                               path.rows()[0].state[2],
                               feature_length};
      auto obs = actor_features(state, xr, previous, meta), error = state_error(state, xr);
      auto proposal =
          supplied ? controller.propose(obs, section(epsilon, 8 * k, 8)) : controller.propose(obs);
      auto warm_before = guess;
      bool success = false, primary = false;
      int attempts = 0;
      double violation = std::numeric_limits<double>::infinity();
      std::vector<double> control = previous;
      for (int attempt = 0; attempt < 2; ++attempt) {
        ++attempts;
        auto attempt_guess =
            attempt == 0 ? guess : rollout_guess(path, ref, config, solver, state, previous, true);
        solver.set_step(state, previous, ref.states, attempt_guess, proposal.q, proposal.r,
                        proposal.p);
        auto result = solver.evaluate_and_solve();
        violation = result.at("max_violation")[0];
        success = result.at("success")[0] == 1;
        if (success) {
          control = result.at("control");
          guess = result.at("shifted");
          primary = attempt == 0;
          break;
        }
      }
      if (success) {
        fail_streak = 0;
        ++successes;
      } else
        ++fail_streak;
      std::vector<double> du{control[0] - previous[0], control[1] - previous[1]};
      double reward = 0;
      for (int j = 0; j < 6; ++j)
        reward -= qe[j] * error[j] * error[j];
      for (int j = 0; j < 2; ++j)
        reward -= re[j] * du[j] * du[j];
      bool invalid = false;
      auto next = state;
      try {
        next = solver.predict(state, control);
      } catch (const std::exception &) {
        invalid = true;
      }
      auto next_ref = tracker.get(next[4], next[5], ts, n);
      auto next_error = state_error(next, section(next_ref.states, 0, 6));
      reached = next_ref.projection.s >= cad_end - .03;
      catastrophic = invalid || fail_streak >= 3;
      terminated = reached || catastrophic;
      truncated = k + 1 >= max_steps && !terminated;
      ensure(mode == Mode::Fixed || !(reached && catastrophic),
             "Both normal/failure termination in adaptive reward");
      double training_reward =
          reward - (success ? 0 : solver_penalty) - (catastrophic ? failure_penalty : 0);
      auto update = controller.observe(
          error, next_error, mode == Mode::Fixed ? reward : training_reward, terminated, truncated);
      auto p_after = mode == Mode::Fixed ? proposal.p : controller.model().at("critic_p");
      std::vector<double> actual_epsilon(8);
      for (int j = 0; j < 8; ++j)
        actual_epsilon[j] =
            (proposal.actor.latent[j] - proposal.actor.mu[j]) / proposal.actor.sigma[j];
      double warm_failure_delta = 0;
      if (!success)
        for (std::size_t j = 0; j < guess.size(); ++j)
          warm_failure_delta = std::max(warm_failure_delta, std::abs(guess[j] - warm_before[j]));
      double nan = std::numeric_limits<double>::quiet_NaN();
      Numbers row{{"state", state},
                  {"next_state", next},
                  {"obs", obs},
                  {"error", error},
                  {"next_error", next_error},
                  {"q", proposal.q},
                  {"r", proposal.r},
                  {"p", proposal.p},
                  {"p_after", p_after},
                  {"mu", proposal.actor.mu},
                  {"sigma", proposal.actor.sigma},
                  {"latent", proposal.actor.latent},
                  {"epsilon", actual_epsilon},
                  {"step", {static_cast<double>(k + 1)}},
                  {"t_s", {k * ts}},
                  {"X_m", {state[4]}},
                  {"Y_m", {state[5]}},
                  {"X_next_m", {next[4]}},
                  {"Y_next_m", {next[5]}},
                  {"X_ref_m", {xr[4]}},
                  {"Y_ref_m", {xr[5]}},
                  {"s_progress_m", {ref.projection.s}},
                  {"progress_m", {next_ref.projection.s}},
                  {"cross_track_m", {ref.projection.distance}},
                  {"e_psi", {error[2]}},
                  {"e_vx", {error[0]}},
                  {"a", {control[0]}},
                  {"steering", {control[1]}},
                  {"du_a", {du[0]}},
                  {"du_delta", {du[1]}},
                  {"reward", {reward}},
                  {"training_reward", {training_reward}},
                  {"td", {update.td}},
                  {"actor_param_step_norm", {update.actor_step_norm}},
                  {"critic_param_step_norm", {update.critic_step_norm}},
                  {"actor_batch_size", {static_cast<double>(update.batch_size)}},
                  {"actor_update_applied", {update.batch_size > 0 ? 1. : 0.}},
                  {"actor_advantage", {nan}},
                  {"actor_target", {nan}},
                  {"actor_advantage_horizon", {0}},
                  {"warm_start_delta_on_failure", {warm_failure_delta}},
                  {"solver_success", {success ? 1. : 0.}},
                  {"solver_primary_success", {primary ? 1. : 0.}},
                  {"solver_attempts", {static_cast<double>(attempts)}},
                  {"max_constraint_violation", {violation}},
                  {"failure_streak", {static_cast<double>(fail_streak)}},
                  {"invalid_plant_state", {invalid ? 1. : 0.}},
                  {"reached_end", {reached ? 1. : 0.}},
                  {"failure_terminal", {catastrophic ? 1. : 0.}},
                  {"terminated", {terminated ? 1. : 0.}},
                  {"truncated", {truncated ? 1. : 0.}}};
      bool ua = mode == Mode::ActorAdapt || mode == Mode::FullAdapt;
      int h = integer(model, "horizon", 1, 10000);
      rows.push_back(row);
      if (ua && h == 1) {
        double v = 0;
        for (int j = 0; j < 6; ++j)
          v -= proposal.p[j] * error[j] * error[j];
        rows.back()["actor_advantage"] = {update.td};
        rows.back()["actor_target"] = {update.td + v};
        rows.back()["actor_advantage_horizon"] = {1};
      }
      if (ua && h > 1) {
        ++pending_count;
        for (std::size_t i = 0; i < update.returns.advantages.size(); ++i) {
          rows.at(pending_start + i)["actor_advantage"] = {update.returns.advantages[i]};
          rows.at(pending_start + i)["actor_target"] = {update.returns.targets[i]};
          rows.at(pending_start + i)["actor_advantage_horizon"] = {update.returns.horizons[i]};
        }
        auto collect =
            static_cast<std::size_t>(integer(model, "design", 0, 1) == 0 ? 2 * h - 1 : h);
        if (pending_count >= collect || terminated || truncated) {
          pending_start = rows.size();
          pending_count = 0;
        }
      }
      total_reward += reward;
      discounted_cost -= discount * reward;
      discount *= scalar(model, "gamma");
      cte_sum += ref.projection.distance;
      cte_max = std::max(cte_max, ref.projection.distance);
      state = next;
      previous = control;
      ref = next_ref;
      if (terminated || truncated)
        break;
    }
    write_csv(argv[4], rows);
    write_numbers(std::string(argv[4]) + ".model", "AC_MODEL_V1", controller.model());
    Numbers summary{{"steps", {static_cast<double>(rows.size())}},
                    {"reached_end", {reached ? 1. : 0.}},
                    {"failure_terminal", {catastrophic ? 1. : 0.}},
                    {"terminated", {terminated ? 1. : 0.}},
                    {"truncated", {truncated ? 1. : 0.}},
                    {"actor_updates", {static_cast<double>(controller.actor_updates())}},
                    {"critic_updates", {static_cast<double>(controller.critic_updates())}},
                    {"actor_training_samples", {static_cast<double>(controller.actor_samples())}},
                    {"total_reward", {total_reward}},
                    {"discounted_cost_from_start", {discounted_cost}},
                    {"mean_cte_all_m", {cte_sum / rows.size()}},
                    {"max_cte_all_m", {cte_max}},
                    {"solver_success_pct", {100. * successes / rows.size()}},
                    {"completion_pct", {100. * rows.back().at("s_progress_m")[0] / cad_end}},
                    {"cad_s_end", {cad_end}},
                    {"feature_length", {feature_length}}};
    write_numbers(std::string(argv[4]) + ".summary", "ROUTE_SUMMARY_V1", summary);
    return catastrophic ? 2 : (reached ? 0 : 3);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
