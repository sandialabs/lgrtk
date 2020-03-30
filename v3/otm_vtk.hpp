#pragma once

#include <otm_host_pinned_state.hpp>
#include <string>

namespace lgr {

class state;

class otm_file_writer
{
  std::string prefix;
public:
  otm_file_writer(std::string const &prefix_in) :
      prefix(prefix_in)
  {
  }
  void capture(state const &s);
  void write(int const file_output_index);

  otm_host_pinned_state host_s;
};

}
