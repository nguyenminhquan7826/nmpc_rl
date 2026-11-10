#include "reference_path.hpp"
#include <algorithm>
#include <limits>
#include <set>
#include <sstream>
namespace deployment {
namespace {
constexpr double pi = 3.14159265358979323846;
double clip(double x, double lo, double hi) { return std::max(lo, std::min(hi, x)); }
std::string trim(const std::string &s) {
  auto a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos)
    return "";
  auto b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}
std::vector<std::string> csv(const std::string &line) {
  std::vector<std::string> result;
  std::string value;
  bool quoted = false, closed = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    char c = line[i];
    if (quoted) {
      if (c == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') {
          value += '"';
          ++i;
        } else {
          quoted = false;
          closed = true;
        }
      } else
        value += c;
    } else if (c == ',') {
      result.push_back(trim(value));
      value.clear();
      closed = false;
    } else if (c == '"') {
      ensure(trim(value).empty() && !closed, "Malformed CSV quote");
      value.clear();
      quoted = true;
    } else {
      ensure(!closed || c == ' ' || c == '\t' || c == '\r', "Characters after CSV quote");
      value += c;
    }
  }
  ensure(!quoted, "Unterminated CSV quote");
  result.push_back(trim(value));
  return result;
}
double number(const std::string &s) {
  ensure(!s.empty(), "Missing CSV value");
  std::size_t used = 0;
  double x = std::stod(s, &used);
  ensure(used == s.size() && std::isfinite(x), "Invalid/nonfinite CSV number");
  return x;
}
bool close(double a, double b, double atol, double rtol) {
  return std::abs(a - b) <= atol + rtol * std::abs(b);
}
} // namespace
ReferencePath ReferencePath::load_csv(const std::string &file) {
  std::ifstream input(file);
  ensure(bool(input), "Cannot open reference CSV: " + file);
  std::string line;
  ensure(bool(std::getline(input, line)), "Empty reference CSV");
  if (line.compare(0, 3, "\xEF\xBB\xBF") == 0)
    line.erase(0, 3);
  auto header = csv(line);
  std::map<std::string, std::size_t> col;
  for (std::size_t i = 0; i < header.size(); ++i)
    ensure(!header[i].empty() && col.emplace(header[i], i).second, "Duplicate/empty CSV header");
  const std::vector<std::string> names{"s_m",         "vx_ref_mps", "vy_ref_mps", "psi_ref_rad",
                                       "r_ref_radps", "X_ref_m",    "Y_ref_m",    "kappa_1_per_m"};
  for (const auto &name : names)
    ensure(col.count(name), "Missing reference column: " + name);
  ReferencePath out;
  out.has_time_ = col.count("t_ref_s");
  out.has_index_ = col.count("index");
  while (std::getline(input, line)) {
    if (trim(line).empty())
      continue;
    ensure(line.size() <= 65536, "Reference CSV line too long");
    auto fields = csv(line);
    ensure(fields.size() == header.size(), "Wrong CSV field count");
    auto get = [&](const std::string &k) { return number(fields[col.at(k)]); };
    RouteRow row;
    row.s = get("s_m");
    for (int j = 0; j < 6; ++j)
      row.state[j] = get(names[j + 1]);
    row.curvature = get("kappa_1_per_m");
    if (out.has_time_)
      row.time = get("t_ref_s");
    if (out.has_index_)
      row.index = get("index");
    if (!out.rows_.empty())
      ensure(row.s > out.rows_.back().s, "s_m must strictly increase");
    out.rows_.push_back(row);
    ensure(out.rows_.size() <= 1000000, "Reference too large");
  }
  ensure(input.eof() && out.rows_.size() >= 2, "Reference needs at least two rows");
  double correction = 0, previous = out.rows_[0].state[2];
  for (std::size_t i = 1; i < out.rows_.size(); ++i) {
    double raw = out.rows_[i].state[2], diff = raw - previous;
    double mod = std::fmod(diff + pi, 2 * pi);
    if (mod < 0)
      mod += 2 * pi;
    mod -= pi;
    if (mod == -pi && diff > 0)
      mod = pi;
    if (std::abs(diff) >= pi)
      correction += mod - diff;
    out.rows_[i].state[2] = raw + correction;
    previous = raw;
  }
  return out;
}
std::pair<std::size_t, double> ReferencePath::interval(double s) const {
  ensure(std::isfinite(s), "Nonfinite reference progress");
  s = clip(s, start(), end());
  auto it = std::upper_bound(rows_.begin(), rows_.end(), s,
                             [](double v, const RouteRow &r) { return v < r.s; });
  std::size_t j = it == rows_.begin() ? 0 : static_cast<std::size_t>(it - rows_.begin() - 1);
  j = std::min(j, rows_.size() - 2);
  return {j, s - rows_[j].s};
}
std::vector<double> ReferencePath::interpolate(double s, bool stop) const {
  auto q = interval(s);
  const auto &a = rows_[q.first];
  const auto &b = rows_[q.first + 1];
  std::vector<double> out(6);
  for (int j = 0; j < 6; ++j)
    out[j] = s >= end() ? rows_.back().state[j]
                        : a.state[j] + ((b.state[j] - a.state[j]) / (b.s - a.s)) * q.second;
  if (stop && s >= end() - 1e-10)
    out[0] = out[1] = out[3] = 0;
  return out;
}
double ReferencePath::curvature(double s) const {
  auto q = interval(s);
  if (s >= end())
    return rows_.back().curvature;
  return rows_[q.first].curvature + ((rows_[q.first + 1].curvature - rows_[q.first].curvature) /
                                     (rows_[q.first + 1].s - rows_[q.first].s)) *
                                        q.second;
}
double ReferencePath::time(double s) const {
  ensure(has_time_, "No t_ref_s column");
  auto q = interval(s);
  if (s >= end())
    return rows_.back().time;
  return rows_[q.first].time + ((rows_[q.first + 1].time - rows_[q.first].time) /
                                (rows_[q.first + 1].s - rows_[q.first].s)) *
                                   q.second;
}
void ReferencePath::validate_cad(double min_speed, double max_speed, double wheelbase,
                                 double limit) const {
  ensure(rows_.size() >= 3 && has_time_ && has_index_,
         "CAD XREF requires >=3 rows, index and t_ref_s");
  ensure(std::isfinite(min_speed) && std::isfinite(max_speed) && min_speed > 0 &&
             max_speed >= min_speed && std::isfinite(wheelbase) && wheelbase > 0 &&
             std::isfinite(limit) && limit > 0,
         "Invalid CAD limits");
  ensure(std::abs(start()) <= 1e-7, "CAD XREF must start at s=0");
  double elapsed = 0;
  for (std::size_t i = 0; i < rows_.size(); ++i) {
    const auto &r = rows_[i];
    double vx = r.state[0];
    ensure(std::abs(r.state[1]) <= 1e-8, "CAD requires zero lateral reference speed");
    ensure(vx >= min_speed - 1e-9 && vx <= max_speed + 1e-9, "CAD reference speed outside limits");
    ensure(close(r.state[3], vx * r.curvature, 1e-4, 1e-4),
           "CAD yaw rate inconsistent with curvature");
    ensure(std::abs(std::atan(wheelbase * r.curvature)) <= limit + 1e-9,
           "CAD steering exceeds limit");
    if (i) {
      const auto &prev = rows_[i - 1];
      double ds = r.s - prev.s,
             chord = std::hypot(r.state[4] - prev.state[4], r.state[5] - prev.state[5]);
      ensure(chord > 1e-9 && chord <= 1.05 * ds, "CAD chord inconsistent with arc length");
      elapsed += ds / (.5 * (vx + prev.state[0]));
    }
    ensure(close(r.time - rows_[0].time, elapsed, 1e-5, 1e-5), "CAD time inconsistent with speed");
  }
}
void ReferencePath::append_tail(double length) {
  ensure(std::isfinite(length) && length > 0, "Invalid tail length");
  std::vector<double> ds;
  for (std::size_t i = 1; i < rows_.size(); ++i)
    ds.push_back(rows_[i].s - rows_[i - 1].s);
  std::sort(ds.begin(), ds.end());
  double median = ds[ds.size() / 2];
  if (ds.size() % 2 == 0)
    median = .5 * (median + ds[ds.size() / 2 - 1]);
  double count = std::max(2., std::ceil(length / median));
  ensure(count <= 1000000 && rows_.size() + count <= 1000000, "Tail too large");
  int n = static_cast<int>(count);
  double step = length / n;
  auto last = rows_.back(), current = last;
  ensure(last.state[0] > 0, "Tail requires positive speed");
  for (int j = 1; j <= n; ++j) {
    double k = last.curvature * (1. - static_cast<double>(j) / n),
           mid = current.state[2] + .5 * k * step;
    current.state[4] += step * std::cos(mid);
    current.state[5] += step * std::sin(mid);
    current.state[2] += k * step;
    current.s += step;
    current.time += step / last.state[0];
    current.curvature = k;
    current.index = last.index + j;
    current.state[1] = 0;
    current.state[3] = last.state[0] * k;
    rows_.push_back(current);
  }
}
RouteTracker::RouteTracker(const ReferencePath &path, bool monotonic, double back, double forward,
                           bool stop, bool midpoint)
    : path_(path), monotonic_(monotonic), stop_(stop), midpoint_(midpoint), back_(back),
      forward_(forward) {
  ensure(std::isfinite(back) && std::isfinite(forward) && back >= 0 && forward > 0,
         "Invalid projection search window");
}
void RouteTracker::reset() { initialized_ = false; }
Projection RouteTracker::project(double x, double y) {
  ensure(std::isfinite(x) && std::isfinite(y), "Nonfinite localization");
  const auto &rows = path_.rows();
  std::vector<std::size_t> eligible;
  double lo = std::max(path_.start(), previous_ - back_),
         hi = std::min(path_.end(), previous_ + forward_);
  for (std::size_t j = 0; j + 1 < rows.size(); ++j)
    if (!initialized_ || (rows[j].s <= hi && rows[j + 1].s >= lo))
      eligible.push_back(j);
  if (eligible.empty())
    for (std::size_t j = 0; j + 1 < rows.size(); ++j)
      eligible.push_back(j);
  double best = std::numeric_limits<double>::infinity();
  Projection out{};
  for (auto j : eligible) {
    const auto &a = rows[j];
    const auto &b = rows[j + 1];
    double dx = b.state[4] - a.state[4], dy = b.state[5] - a.state[5];
    double alpha = clip(
        ((x - a.state[4]) * dx + (y - a.state[5]) * dy) / std::max(dx * dx + dy * dy, 1e-15), 0, 1);
    double xp = a.state[4] + alpha * dx, yp = a.state[5] + alpha * dy,
           d2 = (x - xp) * (x - xp) + (y - yp) * (y - yp);
    if (d2 < best) {
      best = d2;
      out = {a.s + alpha * (b.s - a.s), xp, yp, std::sqrt(d2), j, alpha};
    }
  }
  ensure(std::isfinite(best), "Projection overflow");
  if (initialized_ && monotonic_ && out.s < previous_) {
    out.s = previous_;
    auto r = path_.interpolate(out.s);
    out.x = r[4];
    out.y = r[5];
    out.distance = std::hypot(x - out.x, y - out.y);
  }
  previous_ = out.s;
  initialized_ = true;
  return out;
}
RouteReference RouteTracker::from_progress(double s, double ts, int n) const {
  ensure(std::isfinite(s) && std::isfinite(ts) && ts > 0 && n >= 1 && n <= 1000,
         "Invalid horizon settings");
  RouteReference out{};
  out.progress.push_back(clip(s, path_.start(), path_.end()));
  for (int i = 0; i < n; ++i) {
    double si = out.progress.back();
    auto r = path_.interpolate(si, stop_);
    double v = std::hypot(r[0], r[1]);
    if (midpoint_ && si < path_.end()) {
      auto mid = path_.interpolate(std::min(si + .5 * ts * v, path_.end()), stop_);
      v = std::hypot(mid[0], mid[1]);
    }
    out.progress.push_back(std::min(si + v * ts, path_.end()));
  }
  for (double si : out.progress) {
    extend(out.states, path_.interpolate(si, stop_));
    out.curvatures.push_back(path_.curvature(si));
  }
  return out;
}
RouteReference RouteTracker::get(double x, double y, double ts, int n) {
  auto p = project(x, y);
  auto out = from_progress(p.s, ts, n);
  out.projection = p;
  return out;
}
} // namespace deployment
