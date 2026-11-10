#include "reference_path.hpp"
#include <iostream>
int main(int argc, char **argv) {
  try {
    using namespace deployment;
    ensure(argc == 4, "usage: reference_probe route.csv queries output");
    auto path = ReferencePath::load_csv(argv[1]);
    auto d = read_numbers(argv[2], "ROUTE_QUERY_V1");
    double tail = scalar(d, "tail_length");
    if (tail > 0) {
      path.validate_cad(.05, .1, .2515, .55);
      path.append_tail(tail);
    } else
      ensure(tail == 0, "Invalid tail");
    RouteTracker tracker(path, integer(d, "monotonic", 0, 1), scalar(d, "search_back"),
                         scalar(d, "search_forward"), integer(d, "stop_at_end", 0, 1),
                         integer(d, "midpoint", 0, 1));
    auto xy = d.at("xy");
    ensure(!xy.empty() && xy.size() % 2 == 0, "Invalid XY queries");
    Numbers out;
    for (std::size_t i = 0; i < xy.size(); i += 2) {
      auto r = tracker.get(xy[i], xy[i + 1], scalar(d, "Ts"), integer(d, "N", 1, 1000));
      auto p = r.projection;
      extend(out["projection"],
             {p.s, p.x, p.y, p.distance, static_cast<double>(p.segment), p.alpha});
      extend(out["horizon"], r.progress);
      extend(out["xref"], r.states);
      extend(out["curvature"], r.curvatures);
    }
    for (const auto &r : path.rows()) {
      out["route_rows"].push_back(r.s);
      extend(out["route_rows"], {r.state.begin(), r.state.end()});
      out["route_rows"].push_back(r.curvature);
    }
    write_numbers(argv[3], "ROUTE_RESULT_V1", out);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
