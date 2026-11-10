#include "fixed_nmpc.hpp"
#include "numeric_io.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::invalid_argument("usage: fixed_nmpc_probe input.txt output.txt");
    std::ifstream input(argv[1]);
    std::string version, key;
    if (!(input >> version) || version != "FIXED_NMPC_V1")
      throw std::invalid_argument("Invalid input version");
    deployment::Values fields;
    while (input >> key) {
      long count;
      if (!(input >> count) || count < 0 || count > 100000 || fields.count(key))
        throw std::invalid_argument("Invalid field: " + key);
      std::vector<double> values;
      for (long i = 0; i < count; ++i) {
        std::string token;
        if (!(input >> token))
          throw std::invalid_argument("Truncated field");
        std::size_t used = 0;
        double v = std::stod(token, &used);
        if (used != token.size())
          throw std::invalid_argument("Invalid number");
        values.push_back(v);
      }
      fields.emplace(key, values);
    }
    bool dynamic = fields.count("dynamic_weights")
                       ? deployment::integer(fields, "dynamic_weights", 0, 1) != 0
                       : false;
    deployment::FixedNmpc solver(fields, dynamic);
    auto out = solver.evaluate_and_solve();
    if (out.at("success")[0] != 1)
      std::cerr << "Solver failure: " << solver.status() << "\n";
    std::ofstream output(argv[2]);
    if (!output)
      throw std::runtime_error("Cannot open output");
    output << "FIXED_NMPC_RESULT_V1\n" << std::setprecision(17);
    for (const auto &item : out) {
      output << item.first << " " << item.second.size();
      for (double v : item.second)
        output << " " << v;
      output << "\n";
    }
    output.flush();
    if (!output)
      throw std::runtime_error("Cannot write output");
    return out.at("success")[0] == 1 ? 0 : 2;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
