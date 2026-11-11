#include "fixed_nmpc.hpp"
#include "online_controller.hpp"
#include <iostream>
int main(int argc, char **argv) {
  try {
    using namespace deployment;
    ensure(argc == 5 || argc == 6, "usage: four_mode_probe model scenario output mode [seed]");
    auto model = read_numbers(argv[1], "AC_MODEL_V1"),
         d = read_numbers(argv[2], "AC_ROLLOUT_INPUT_V1");
    const std::map<std::string, Mode> modes{{"fixed", Mode::Fixed},
                                            {"actor_adapt", Mode::ActorAdapt},
                                            {"critic_adapt", Mode::CriticAdapt},
                                            {"full_adapt", Mode::FullAdapt}};
    ensure(modes.count(argv[4]), "Unknown controller mode");
    auto mode = modes.at(argv[4]);
    unsigned long long seed = 0;
    if (argc == 6) {
      std::string text = argv[5];
      ensure(!text.empty() && text.find_first_not_of("0123456789") == std::string::npos,
             "Invalid seed");
      std::size_t used;
      seed = std::stoull(text, &used);
      ensure(used == text.size(), "Invalid seed");
    }
    OnlineController controller(model, mode, seed);
    int steps = integer(d, "steps", 1, 100000), N = integer(d, "N", 1, 1000);
    auto refs = field(d, "reference_sequence", 6 * (steps + N + 1)),
         metas = field(d, "feature_meta_sequence", 9 * steps);
    auto qe = field(d, "task_q", 6), re = field(d, "task_r", 2);
    for (double v : qe)
      ensure(v > 0, "Invalid task Qe");
    for (double v : re)
      ensure(v > 0, "Invalid task Re");
    bool terminal = integer(d, "terminal", 0, 1) != 0;
    bool supplied = d.count("epsilon");
    std::vector<double> eps;
    if (supplied)
      eps = field(d, "epsilon", steps * 8);
    d["q"] = model.at("fixed_q");
    d["r"] = model.at("fixed_r");
    d["qf"] = model.at("fixed_qf");
    FixedNmpc solver(d, true);
    auto state = field(d, "state", 6), previous = field(d, "previous", 2),
         guess = field(d, "guess", 8 * N);
    Numbers out;
    bool failed = false;
    for (int i = 0; i < steps; ++i) {
      auto xr = section(refs, i * 6, 6), meta = section(metas, i * 9, 9);
      ensure(meta[8] == scalar(model, "train_length"), "Feature train length differs from model");
      auto obs = actor_features(state, xr, previous, meta), err = state_error(state, xr);
      auto p = supplied ? controller.propose(obs, section(eps, i * 8, 8)) : controller.propose(obs);
      solver.set_step(state, previous, section(refs, i * 6, 6 * (N + 1)), guess, p.q, p.r, p.p);
      auto result = solver.evaluate_and_solve();
      if (result.at("success")[0] != 1) {
        controller.cancel();
        std::cerr << "Solve failed at step " << i << ": " << solver.status() << '\n';
        failed = true;
        break;
      }
      auto control = result.at("control"), next = solver.predict(state, control);
      auto en = state_error(next, section(refs, (i + 1) * 6, 6));
      double reward = 0;
      for (int j = 0; j < 6; ++j)
        reward -= qe[j] * err[j] * err[j];
      for (int j = 0; j < 2; ++j) {
        double du = control[j] - previous[j];
        reward -= re[j] * du * du;
      }
      auto update = controller.observe(err, en, reward, terminal && i == steps - 1,
                                       !terminal && i == steps - 1);
      extend(out["states"], state);
      extend(out["next_states"], next);
      extend(out["controls"], control);
      extend(out["weights"], p.q);
      extend(out["weights"], p.r);
      extend(out["p_before"], p.p);
      extend(out["p_after"], mode == Mode::Fixed ? p.p : controller.model().at("critic_p"));
      extend(out["mu"], p.actor.mu);
      extend(out["sigma"], p.actor.sigma);
      extend(out["latent"], p.actor.latent);
      std::vector<double> actual_epsilon(8);
      for (int j = 0; j < 8; ++j)
        actual_epsilon[j] = (p.actor.latent[j] - p.actor.mu[j]) / p.actor.sigma[j];
      extend(out["epsilon"], actual_epsilon);
      out["reward"].push_back(reward);
      out["td"].push_back(update.td);
      out["batch_size"].push_back(static_cast<double>(update.batch_size));
      out["actor_step_norm"].push_back(update.actor_step_norm);
      out["critic_step_norm"].push_back(update.critic_step_norm);
      out["max_violation"].push_back(result.at("max_violation")[0]);
      state = next;
      previous = control;
      guess = result.at("shifted");
    }
    auto final = controller.model();
    for (const auto &key : {"W1", "b1", "W2", "b2", "W_mu", "b_mu", "W_sigma", "b_sigma"})
      out[std::string("final_") + key] = final.at(key);
    out["actor_updates"] = {static_cast<double>(controller.actor_updates())};
    out["critic_updates"] = {static_cast<double>(controller.critic_updates())};
    out["actor_samples"] = {static_cast<double>(controller.actor_samples())};
    out["complete"] = {failed ? 0. : 1.};
    out["seed"] = {static_cast<double>(seed)};
    write_numbers(argv[3], "AC_ROLLOUT_V1", out);
    write_numbers(std::string(argv[3]) + ".model", "AC_MODEL_V1", final);
    return failed ? 2 : 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
