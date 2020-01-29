#include <ArborX_Point.hpp>
#include <hpc_array_vector.hpp>
#include <hpc_range.hpp>
#include <hpc_vector3.hpp>
#include <Kokkos_Core.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <otm_arborx_search_impl.hpp>

namespace lgr {
namespace search {
namespace impl {

using device_range_type = Kokkos::RangePolicy<device_exec_space>;
using Kokkos::parallel_for;

device_point_view create_arborx_nodes(const lgr::state& s)
{
  device_point_view search_nodes("nodes", s.nodes.size());

  auto nodes_to_x = s.x.cbegin();
  device_range_type node_range(0, search_nodes.extent(0));
  parallel_for(node_range, KOKKOS_LAMBDA(int node_idx)
  {
    using NI = lgr::node_index;
    auto&& search_node = search_nodes(node_idx);
    auto&& lgr_node_coord = nodes_to_x[NI(node_idx)].load();
    search_node[0] = lgr_node_coord(0);
    search_node[1] = lgr_node_coord(1);
    search_node[2] = lgr_node_coord(2);
  });

  return search_nodes;
}

device_point_view create_arborx_points(const lgr::state& s)
{
  device_point_view search_points("points", s.points.size());

  auto mat_pts_to_x = s.xm.cbegin();
  device_range_type pt_range(0, search_points.extent(0));
  parallel_for(pt_range, KOKKOS_LAMBDA(int pt_idx)
  {
    using PI = lgr::point_index;
    auto&& search_pt = search_points(pt_idx);
    auto&& lgr_pt_coord = mat_pts_to_x[PI(pt_idx)].load();
    search_pt[0] = lgr_pt_coord(0);
    search_pt[1] = lgr_pt_coord(1);
    search_pt[2] = lgr_pt_coord(2);
  });

  return search_points;
}

}
}
}
