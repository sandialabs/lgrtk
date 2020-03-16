#include <ArborX_LinearBVH.hpp>
#include <ArborX_Predicates.hpp>
#include <hpc_array.hpp>
#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_macros.hpp>
#include <hpc_range.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Pair.hpp>
#include <otm_state.hpp>
#include <otm_arborx_search_impl.hpp>
#include <string>

namespace lgr {
namespace search {
namespace arborx {

using device_range = Kokkos::RangePolicy<device_exec_space>;
using device_bvh = ArborX::BoundingVolumeHierarchy<lgr::search::arborx::device_mem_space>;

using Kokkos::parallel_for;
using Kokkos::fence;
using Kokkos::tie;

template<typename query_view_type>
HPC_NOINLINE void do_search(device_point_view nodes, query_view_type queries,
    device_int_view& indices, device_int_view& offsets)
{
  device_bvh bvh(nodes);
  bvh.query(queries, indices, offsets);
}

HPC_NOINLINE device_nearest_query_view make_nearest_node_queries(device_point_view points,
    const int num_nodes_to_find)
{
  const int numQueries = points.extent(0);
  device_nearest_query_view queries(Kokkos::ViewAllocateWithoutInitializing("queries"), numQueries);

  parallel_for("setup_queries", device_range(0, numQueries),
  KOKKOS_LAMBDA(int i)
  {
    queries(i) = ArborX::nearest(points(i), num_nodes_to_find);
  });
  fence();
  return queries;
}

HPC_NOINLINE device_intersects_query_view make_intersect_sphere_queries(device_sphere_view point_spheres)
{
  const int numQueries = point_spheres.extent(0);
  device_intersects_query_view queries(Kokkos::ViewAllocateWithoutInitializing("queries"), numQueries);

  parallel_for("setup_queries", device_range(0, numQueries),
  KOKKOS_LAMBDA(int i)
  {
    queries(i) = ArborX::intersects(point_spheres(i));
  });
  fence();
  return queries;
}

template<typename idx_type>
HPC_NOINLINE device_point_view make_point_view(const std::string &view_name,
    const hpc::counting_range<idx_type> &lgr_points,
    const hpc::device_array_vector<hpc::position<double>, idx_type> &coords)
{
  device_point_view search_points(view_name, lgr_points.size());
  auto points_to_x = coords.cbegin();
  device_range point_range(0, search_points.extent(0));
  parallel_for(point_range, KOKKOS_LAMBDA(int i)
  {
    auto&& search_node = search_points(i);
    auto&& lgr_node_coord = points_to_x[idx_type(i)].load();
    search_node[0] = lgr_node_coord(0);
    search_node[1] = lgr_node_coord(1);
    search_node[2] = lgr_node_coord(2);
  });
  fence();
  return search_points;
}

template<typename idx_type>
HPC_NOINLINE device_sphere_view make_sphere_view(const std::string &view_name,
    const hpc::counting_range<idx_type> &lgr_points,
    const hpc::device_array_vector<hpc::position<double>, idx_type> &coords,
    const hpc::device_vector<hpc::length<double>, idx_type> &radii)
{
  using sphere = ArborX::Sphere;
  device_sphere_view search_spheres(view_name, lgr_points.size());
  auto points_to_x = coords.cbegin();
  auto points_to_r = radii.cbegin();
  device_range point_range(0, search_spheres.extent(0));
  parallel_for(point_range, KOKKOS_LAMBDA(int i)
  {
    auto&& search_sphere = search_spheres(i);
    auto&& coords = points_to_x[idx_type(i)].load();
    auto&& radius = points_to_r[idx_type(i)];
    search_sphere = sphere({ coords(0), coords(1), coords(2)}, radius);
  });
  fence();
  return search_spheres;
}

HPC_NOINLINE device_point_view create_arborx_nodes(const lgr::state& s)
{
  return make_point_view("nodes", s.nodes, s.x);
}

HPC_NOINLINE device_point_view create_arborx_points(const lgr::state& s)
{
  return make_point_view("points", s.points, s.xp);
}

HPC_NOINLINE device_sphere_view create_arborx_point_spheres(const lgr::state& s)
{
  return make_sphere_view("point_spheres", s.points, s.xp, s.h_otm);
}

void inflate_sphere_query_radii(device_intersects_query_view queries, double factor)
{
  device_range r(0, queries.extent(0));
  parallel_for(r, KOKKOS_LAMBDA(int i)
  {
    auto&& query = queries(i);
    auto&& sphere = query._geometry;
    sphere._radius *= factor;
  });
}

void initialize()
{
  Kokkos::initialize();
}

void finalize()
{
  Kokkos::finalize();
}

template void do_search(device_point_view nodes, device_nearest_query_view queries, device_int_view& indices, device_int_view& offsets);
template void do_search(device_point_view nodes, device_intersects_query_view queries, device_int_view& indices, device_int_view& offsets);

}
}
}

