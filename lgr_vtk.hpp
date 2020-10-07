#pragma once

#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <lgr_mesh_indices.hpp>
#include <string>

namespace lgr {

class input;
class state;

class captured_state
{
 public:
  hpc::counting_range<element_index>                                                             elements{0};
  hpc::counting_range<node_index>                                                                nodes{0};
  hpc::counting_range<node_in_element_index>                                                     nodes_in_element{0};
  hpc::counting_range<point_in_element_index>                                                    points_in_element{0};
  hpc::pinned_vector<node_index, element_node_index>                                             element_nodes_to_nodes;
  hpc::pinned_array_vector<hpc::position<double>, node_index>                                    x;
  hpc::pinned_array_vector<hpc::velocity<double>, node_index>                                    v;
  hpc::host_vector<hpc::pinned_vector<hpc::pressure<double>, node_index>, material_index>        p_h;
  hpc::host_vector<hpc::pinned_vector<hpc::specific_energy<double>, node_index>, material_index> e_h;
  hpc::host_vector<hpc::pinned_vector<hpc::density<double>, node_index>, material_index>         rho_h;
  hpc::pinned_vector<hpc::length<double>, node_index>                                            h_adapt;
  hpc::pinned_vector<hpc::pressure<double>, point_index>                                         p;
  hpc::pinned_vector<hpc::specific_energy<double>, point_index>                                  e;
  hpc::pinned_vector<hpc::density<double>, point_index>                                          rho;
  hpc::pinned_vector<hpc::density<double>, point_index>                                          ep;
  hpc::pinned_array_vector<hpc::heat_flux<double>, point_index>                                  q;
  hpc::pinned_vector<hpc::pressure<double>, point_index>                                         p_prime;
  hpc::pinned_vector<hpc::time<double>, point_index>                                             element_dt;
  hpc::pinned_vector<hpc::adimensional<double>, element_index>                                   quality;
  hpc::pinned_vector<material_index, element_index>                                              material;
};

class file_writer
{
  std::string prefix;

 public:
  file_writer(std::string const& prefix_in) : prefix(prefix_in)
  {
  }
  void
  capture(input const& in, state const& s);
  void
                 write(input const& in, int const file_output_index);
  captured_state captured;
};

}  // namespace lgr
