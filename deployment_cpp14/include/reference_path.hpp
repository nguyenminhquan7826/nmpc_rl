#pragma once
#include "numeric_io.hpp"
#include <array>
namespace deployment {
struct RouteRow {
  double s;
  std::array<double, 6> state;
  double curvature, time = 0, index = 0;
};
class ReferencePath {
public:
  static ReferencePath load_csv(const std::string &file);
  void validate_cad(double min_speed, double max_speed, double wheelbase,
                    double steering_limit) const;
  void append_tail(double length);
  std::vector<double> interpolate(double s, bool stop_at_end = false) const;
  double curvature(double s) const;
  double time(double s) const;
  const std::vector<RouteRow> &rows() const { return rows_; }
  double start() const { return rows_.front().s; }
  double end() const { return rows_.back().s; }
  bool has_time() const { return has_time_; }

private:
  std::vector<RouteRow> rows_;
  bool has_time_ = false, has_index_ = false;
  std::pair<std::size_t, double> interval(double s) const;
};
struct Projection {
  double s, x, y, distance;
  std::size_t segment;
  double alpha;
};
struct RouteReference {
  Projection projection;
  std::vector<double> progress, states, curvatures;
};
class RouteTracker {
public:
  explicit RouteTracker(const ReferencePath &path, bool monotonic = true, double back = .15,
                        double forward = 1.2, bool stop = false, bool midpoint = true);
  void reset();
  Projection project(double x, double y);
  RouteReference from_progress(double s, double ts, int n) const;
  RouteReference get(double x, double y, double ts, int n);

private:
  const ReferencePath &path_;
  bool monotonic_, stop_, midpoint_, initialized_ = false;
  double back_, forward_, previous_ = 0;
};
} // namespace deployment
