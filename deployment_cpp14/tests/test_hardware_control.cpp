#include "hardware_control.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace deployment;
int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--emit-commands") {
    std::cout << command_frame(1, "ARM", 0, 75)
              << command_frame(2, "RUN", .1, 90)
              << command_frame(3, "STOP", 0, 75);
    return 0;
  }
  if (argc == 2 && std::string(argv[1]) == "--verify-acks") {
    std::string line;
    unsigned count = 0;
    while (std::getline(std::cin, line)) {
      Ack a;
      assert(parse_ack(line, a) && a.reason == 0);
      assert(a.armed == (count < 2));
      count++;
    }
    assert(count == 3);
    return 0;
  }
  HardwareLimits l;
  auto c = hardware_command(.1, .3962, 0, .05, .1962, l);
  assert(std::abs(c.speed - .11) < 1e-10 && c.servo == 75 && !c.clipped);
  auto left = hardware_command(.1, .1962, 22.835397 * pi / 180, .05, .1962, l);
  assert(left.servo == 105);
  auto right =
      hardware_command(.1, .1962, -22.835397 * pi / 180, .05, .1962, l);
  assert(right.servo == 45);
  l.max_steering = 5.5 * pi / 180;
  auto quant = hardware_command(.1, .1962, l.max_steering, .05, .1962, l);
  assert(std::abs(quant.steering) <= l.max_steering + 1e-10);
  l = HardwareLimits{};
  double previous = wheel_angle(46), upper = previous + .04;
  auto rate =
      hardware_command(.1, .1962, upper, .05, .1962, l, previous - .04, upper);
  assert(rate.steering - previous <= .04 + 1e-10 && rate.servo != 51);
  auto clip = hardware_command(.149, 2, 0, .05, 0, l);
  assert(clip.clipped && clip.speed == l.max_speed);
  bool fail = false;
  try {
    hardware_command(.1, NAN, 0, .05, 0, l);
  } catch (...) {
    fail = true;
  }
  assert(fail);
  auto f = command_frame(3, "RUN", c.speed, c.servo);
  assert(f.find("CTL,1,3,RUN,0.110000,75,") == 0);
  Ack a;
  std::string payload = "ACK,1,3,1,0,120,3";
  assert(parse_ack(checked_frame(payload), a) && a.seq == 3 && a.armed &&
         a.last_seq == 3);
  auto bad = checked_frame(payload);
  bad[6] = '9';
  assert(!parse_ack(bad, a));
  std::cout << "Hardware conversion/calibration and CTL/ACK CRC PASS\n";
}
