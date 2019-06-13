#pragma once

#include <string>

namespace lgr {

class input;
class state;

class file_writer {
  std::string prefix;
  public:
  file_writer(std::string const& prefix_in)
    :prefix(prefix_in)
  {}
  void write(
      input const& in,
      int const file_output_index,
      state const& s
      );
};

}
