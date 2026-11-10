#pragma once
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
namespace deployment {
using Numbers = std::map<std::string, std::vector<double>>;
inline void ensure(bool ok, const std::string &why) {
  if (!ok)
    throw std::invalid_argument(why);
}
inline const std::vector<double> &field(const Numbers &data, const std::string &key,
                                        std::size_t n) {
  auto it = data.find(key);
  ensure(it != data.end() && it->second.size() == n, "Invalid shape: " + key);
  for (double v : it->second)
    ensure(std::isfinite(v), "Nonfinite: " + key);
  return it->second;
}
inline double scalar(const Numbers &data, const std::string &key) { return field(data, key, 1)[0]; }
inline int integer(const Numbers &data, const std::string &key, int lo, int hi) {
  double x = scalar(data, key);
  ensure(x >= lo && x <= hi && std::floor(x) == x, "Invalid integer: " + key);
  return static_cast<int>(x);
}
inline Numbers read_numbers(const std::string &path, const std::string &version) {
  std::ifstream in(path);
  std::string tag, key;
  ensure(bool(in >> tag) && tag == version, "Invalid file/version: " + path);
  Numbers data;
  std::size_t total = 0;
  while (in >> key) {
    long count;
    ensure(bool(in >> count) && count >= 0 && count <= 1000000 && !data.count(key),
           "Invalid field: " + key);
    total += static_cast<std::size_t>(count);
    ensure(total <= 4000000, "Input too large");
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(count));
    for (long i = 0; i < count; ++i) {
      std::string t;
      ensure(bool(in >> t), "Truncated field: " + key);
      std::size_t used = 0;
      double v = std::stod(t, &used);
      ensure(used == t.size(), "Invalid number");
      values.push_back(v);
    }
    data.emplace(key, std::move(values));
  }
  ensure(in.eof(), "Input read error");
  return data;
}
inline void write_numbers(const std::string &path, const std::string &version,
                          const Numbers &data) {
  std::ofstream out(path);
  ensure(bool(out), "Cannot open output: " + path);
  out << version << '\n' << std::setprecision(17);
  for (const auto &entry : data) {
    out << entry.first << ' ' << entry.second.size();
    for (double v : entry.second)
      out << ' ' << v;
    out << '\n';
  }
  out.flush();
  ensure(bool(out), "Output write failed");
}
inline std::vector<double> section(const std::vector<double> &v, std::size_t start, std::size_t n) {
  ensure(start <= v.size() && n <= v.size() - start, "Invalid vector slice");
  return {v.begin() + start, v.begin() + start + n};
}
inline void extend(std::vector<double> &out, const std::vector<double> &values) {
  out.insert(out.end(), values.begin(), values.end());
}
} // namespace deployment
