#pragma once

#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_range.hpp>
#include <lgr_mesh_indices.hpp>
#include <otm_host_pinned_state.hpp>
#include <string>

namespace lgr {

class state;

struct otm_host_pinned_output_state : public otm_host_pinned_state {
  hpc::counting_range<node_index> nodes{node_index(0)};
  hpc::counting_range<point_index> points{point_index(0)};

  hpc::time<double> time;

  hpc::pinned_array_vector<hpc::displacement<double>, node_index> u;
  hpc::pinned_array_vector<hpc::velocity<double>, node_index> v;
  hpc::pinned_vector<hpc::mass<double>, node_index> mass;

  hpc::pinned_array_vector<hpc::deformation_gradient<double>, point_index> F_total;
  hpc::pinned_array_vector<hpc::stress<double>, point_index> sigma;
  hpc::pinned_vector<hpc::pressure<double>, point_index> K;
  hpc::pinned_vector<hpc::pressure<double>, point_index> G;
};

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
  void to_console();

  otm_host_pinned_output_state host_s;
};

}
