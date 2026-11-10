#include "online_controller.hpp"
#include <cassert>
#include <iostream>
#include <limits>
int main(int argc, char **argv) {
  using namespace deployment;
  assert(argc <= 2);
  Numbers m;
  if (argc == 2)
    m = read_numbers(argv[1], "AC_MODEL_V1");
  else {
    for (const auto &t : std::map<std::string, int>{{"W1", 1088},
                                                    {"b1", 64},
                                                    {"W2", 4096},
                                                    {"b2", 64},
                                                    {"W_mu", 512},
                                                    {"b_mu", 8},
                                                    {"W_sigma", 512},
                                                    {"b_sigma", 8}})
      m[t.first] = std::vector<double>(t.second, 0);
    m["lower"] = std::vector<double>(8, .1);
    m["upper"] = std::vector<double>(8, 10);
    m["sigma_bounds"] = {.05, .6};
    for (const auto &k : {"critic_p", "fixed_q", "fixed_qf"})
      m[k] = std::vector<double>(6, 1);
    m["critic_lower"] = std::vector<double>(6, .1);
    m["critic_upper"] = std::vector<double>(6, 10);
    m["fixed_r"] = {1, 1};
    m["critic_rates"] = std::vector<double>(6, 1);
    m["actor_factors"] = std::vector<double>(8, 1);
    m["gamma"] = {.91};
    m["actor_lr"] = {.001};
    m["mu_reg"] = {0, 2};
    m["critic_mode"] = {0};
    m["horizon"] = {20};
    m["design"] = {0};
    m["train_length"] = {10};
  }
  std::vector<double> obs(17, .1), eps(8, .2), e(6, .01), en(6, .02);
  double nan = std::numeric_limits<double>::quiet_NaN();
  OnlineController c(m, Mode::FullAdapt, 2), reference = c;
  auto original = c.model();
  bool threw = false;
  auto bad = obs;
  bad[0] = nan;
  try {
    c.propose(bad);
  } catch (const std::exception &) {
    threw = true;
  }
  assert(threw);
  auto p = c.propose(obs), pr = reference.propose(obs);
  assert(p.actor.latent == pr.actor.latent);
  threw = false;
  try {
    c.propose(obs);
  } catch (const std::exception &) {
    threw = true;
  }
  assert(threw);
  threw = false;
  try {
    c.observe(e, en, nan, false, false);
  } catch (const std::exception &) {
    threw = true;
  }
  assert(threw);
  assert(c.model() == original && c.actor_updates() == 0 && c.critic_updates() == 0);
  c.observe(e, en, -.1, false, false);
  reference.observe(e, en, -.1, false, false);
  assert(c.model() == reference.model());
  c.propose(obs, eps);
  auto before = c.model();
  c.cancel();
  assert(c.model() == before);
  c.propose(obs, eps);
  c.observe(e, en, -.1, false, true);
  threw = false;
  try {
    c.propose(obs);
  } catch (const std::exception &) {
    threw = true;
  }
  assert(threw);
  Actor actor(m);
  Numbers huge;
  for (const auto &t : actor.tensors())
    huge[t.first] = std::vector<double>(t.second.size(), 1e308);
  ActorGradient g{0, huge};
  auto prior = actor.tensors();
  threw = false;
  try {
    actor.apply(g);
  } catch (const std::exception &) {
    threw = true;
  }
  assert(threw && actor.tensors() == prior);
  std::cout << "Online lifecycle, RNG rollback, candidate atomicity and terminal guards PASS\n";
}
