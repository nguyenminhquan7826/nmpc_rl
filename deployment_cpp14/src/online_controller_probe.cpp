#include "online_controller.hpp"
#include <iostream>
int main(int argc, char **argv) {
  try {
    using namespace deployment;
    ensure(argc == 3, "usage: online_controller_probe input output");
    auto d = read_numbers(argv[1], "AC_SEQUENCE_V1");
    auto mode = static_cast<Mode>(integer(d, "mode", 0, 3));
    OnlineController controller(d, mode, 0);
    std::size_t n = d.at("rewards").size();
    ensure(n > 0, "Empty sequence");
    auto obs = field(d, "observations", n * 17), eps = field(d, "epsilon", n * 8),
         errors = field(d, "errors", n * 6), en = field(d, "next_errors", n * 6),
         rewards = field(d, "rewards", n);
    bool terminal = integer(d, "terminal", 0, 1) != 0;
    Numbers out;
    for (std::size_t i = 0; i < n; ++i) {
      auto p = controller.propose(section(obs, i * 17, 17), section(eps, i * 8, 8));
      extend(out["weights"], p.q);
      extend(out["weights"], p.r);
      extend(out["p_before"], p.p);
      auto u = controller.observe(section(errors, i * 6, 6), section(en, i * 6, 6), rewards[i],
                                  terminal && i == n - 1, !terminal && i == n - 1);
      out["td"].push_back(u.td);
      out["batch_size"].push_back(static_cast<double>(u.batch_size));
      extend(out["p_after"],
             mode == Mode::Fixed ? d.at("fixed_qf") : controller.model().at("critic_p"));
    }
    auto model = controller.model();
    for (const auto &name : {"W1", "b1", "W2", "b2", "W_mu", "b_mu", "W_sigma", "b_sigma"})
      out[std::string("final_") + name] = model.at(name);
    out["actor_updates"] = {static_cast<double>(controller.actor_updates())};
    out["actor_samples"] = {static_cast<double>(controller.actor_samples())};
    out["critic_updates"] = {static_cast<double>(controller.critic_updates())};
    write_numbers(argv[2], "AC_RESULT_V1", out);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
