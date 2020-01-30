#include <ArborX_LinearBVH.hpp>
#include <ArborX_Predicates.hpp>
#include <hpc_array.hpp>
#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_macros.hpp>
#include <hpc_range.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>
#include <lgr_state.hpp>
#include <otm_arborx_search_impl.hpp>
#include <string>

using Kokkos::tie;

namespace lgr {
namespace search {
namespace arborx {

using device_range = Kokkos::RangePolicy<device_exec_space>;
using device_bvh = ArborX::BoundingVolumeHierarchy<lgr::search::arborx::device_exec_space>;

using Kokkos::parallel_for;
using Kokkos::fence;

HPC_NOINLINE device_search_results do_search(device_point_view nodes,
    device_nearest_query_view queries)
{
  device_bvh bvh(nodes);
  device_int_view offsets("offsets", 0);
  device_int_view indices("indices", 0);
  bvh.query(queries, indices, offsets);
  return tie(offsets, indices);
}

HPC_NOINLINE device_nearest_query_view make_nearest_node_queries(device_point_view points,
    const int num_nodes_to_find)
{
  const int numQueries = points.extent(0);
  device_nearest_query_view queries(
      Kokkos::ViewAllocateWithoutInitializing("queries"), numQueries);

  parallel_for("setup_queries", device_range(0, numQueries),
  KOKKOS_LAMBDA(int i)
  {
    queries(i) = ArborX::nearest(points(i), num_nodes_to_find);
  });
  fence();
  return queries;
}

template<typename idx_type>
HPC_NOINLINE device_point_view make_point_view(
    const std::string& view_name,
    const hpc::counting_range<idx_type>& lgr_points,
    const hpc::device_array_vector<hpc::position<double>, idx_type>& coords)
{
  device_point_view search_points(view_name, lgr_points.size());
  auto points_to_x = coords.cbegin();
  device_range point_range(0, search_points.extent(0));
  parallel_for(point_range, KOKKOS_LAMBDA(int node_idx)
  {
    auto&& search_node = search_points(node_idx);
    auto&& lgr_node_coord = points_to_x[idx_type(node_idx)].load();
    search_node[0] = lgr_node_coord(0);
    search_node[1] = lgr_node_coord(1);
    search_node[2] = lgr_node_coord(2);
  });
  fence();
  return search_points;
}

HPC_NOINLINE device_point_view create_arborx_nodes(const lgr::state& s)
{
  return make_point_view("nodes", s.nodes, s.x);
}

HPC_NOINLINE device_point_view create_arborx_points(const lgr::state& s)
{
  return make_point_view("points", s.points, s.xm);
}

}
}
}
