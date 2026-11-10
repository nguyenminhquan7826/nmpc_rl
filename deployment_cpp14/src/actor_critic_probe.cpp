#include "online_ac.hpp"
#include <iostream>
int main(int argc, char **argv) {
  try {
    deployment::ensure(argc == 3, "usage: actor_critic_probe input output");
    using namespace deployment;
    auto d = read_numbers(argv[1], "AC_MATH_V1");
    Actor actor(d);
    Critic critic(d);
    Numbers out;
    auto adv = d.at("advantages");
    std::size_t n = adv.size();
    auto obs = field(d, "observations", n * 17), eps = field(d, "epsilon", n * 8),
         z = field(d, "latents", n * 8);
    for (std::size_t i = 0; i < n; ++i) {
      auto p = actor.propose(section(obs, i * 17, 17), section(eps, i * 8, 8));
      extend(out["mu"], p.mu);
      extend(out["sigma"], p.sigma);
      extend(out["latent"], p.latent);
      extend(out["weights"], p.weights);
      out["logp"].push_back(p.log_probability);
    }
    auto grad = actor.gradients(obs, z, adv);
    out["loss"] = {grad.loss};
    for (const auto &t : grad.tensors)
      out["gradient_" + t.first] = t.second;
    actor.apply(grad);
    for (const auto &t : actor.tensors())
      out["after_" + t.first] = t.second;
    bool terminal = integer(d, "terminal", 0, 1) != 0;
    auto step = critic.evaluate(field(d, "error", 6), field(d, "next_error", 6),
                                scalar(d, "reward"), terminal);
    critic.apply(step);
    out["critic_td"] = {step.td};
    out["critic_gradient"] = step.gradient;
    out["critic_after"] = critic.p();
    out["critic_loss"] = {step.loss};
    auto rewards = field(d, "rewards", n), errors = field(d, "errors", n * 6);
    for (bool full : {false, true}) {
      auto r = advantages(rewards, errors, d.at("next_error"), d.at("critic_p"), scalar(d, "gamma"),
                          terminal, 3, full);
      std::string prefix = full ? "full_" : "block_";
      out[prefix + "advantages"] = r.advantages;
      out[prefix + "targets"] = r.targets;
      if (full)
        out["full_horizons"] = r.horizons;
    }
    out["features"] = actor_features(field(d, "feature_state", 6), field(d, "feature_reference", 6),
                                     field(d, "feature_previous", 2), field(d, "feature_meta", 9));
    write_numbers(argv[2], "AC_RESULT_V1", out);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
