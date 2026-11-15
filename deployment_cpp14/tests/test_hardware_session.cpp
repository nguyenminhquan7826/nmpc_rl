#include "hardware_session.hpp"
#include <cassert>
#include <iostream>
using namespace deployment;
int main(int argc, char **argv) {
  assert(argc == 4);
  auto m = read_numbers(argv[1], "AC_MODEL_V1"),
       c = read_numbers(argv[3], "NMPC_HARDWARE_V1");
  auto path = ReferencePath::load_csv(argv[2]);
  for (Mode mode :
       {Mode::Fixed, Mode::ActorAdapt, Mode::CriticAdapt, Mode::FullAdapt}) {
    HardwareSession s(m, c, path, mode, 0);
    Pose p;
    p.vx = .1;
    auto before = s.model();
    auto proposal = s.propose(p);
    assert(proposal.state[1] == 0 && proposal.q.size() == 6);
    s.cancel();
    assert(s.model() == before);
    assert(s.actor_updates() == 0 && s.critic_updates() == 0);
    proposal = s.propose(p);
    s.accept();
    p.x = .008;
    auto update = s.feedback(p, false, false);
    (void)update;
    assert(s.actor_updates() ==
           ((mode == Mode::ActorAdapt || mode == Mode::FullAdapt) ? 1U : 0U));
    assert(s.critic_updates() ==
           ((mode == Mode::CriticAdapt || mode == Mode::FullAdapt) ? 1U : 0U));
  }
  bool fail = false;
  try {
    auto bad = c;
    bad["input_upper"][1] = .55;
    HardwareSession s(m, bad, path, Mode::Fixed, 0);
  } catch (...) {
    fail = true;
  }
  assert(fail);
  std::cout << "Hardware session four modes: shadow cancels, accepted "
               "transition updates, limits PASS\n";
}
