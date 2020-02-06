#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_View.hpp>

#include <ArborX_Point.hpp>
#include <ArborX_Predicates.hpp>

namespace lgr {
class state;
}

namespace ArborX {
class Box;
class Point;
class Sphere;
template <typename GeomType> class Nearest;
template <typename GeomType> class Intersects;
}

namespace lgr {
namespace search {
namespace arborx {

using device_exec_space = Kokkos::DefaultExecutionSpace;
#ifdef KOKKOS_ENABLE_CUDA
using device_mem_space = Kokkos::CudaSpace;
#else
using device_mem_space = device_exec_space::memory_space;
#endif
using device_point_view = Kokkos::View<ArborX::Point *, device_mem_space>;
using device_sphere_view = Kokkos::View<ArborX::Sphere *, device_mem_space>;
using device_nearest_query_view = Kokkos::View<ArborX::Nearest<ArborX::Point> *, device_mem_space>;
using device_intersects_query_view = Kokkos::View<ArborX::Intersects<ArborX::Sphere> *, device_mem_space>;
using device_int_view = Kokkos::View<int *, device_mem_space>;
using device_search_results = Kokkos::pair<device_int_view, device_int_view>;

device_point_view create_arborx_nodes(const lgr::state &s);

device_point_view create_arborx_points(const lgr::state &s);

device_sphere_view create_arborx_point_spheres(const lgr::state &s);

device_nearest_query_view make_nearest_node_queries(device_point_view points,
    const int num_nodes_to_find);

device_intersects_query_view make_intersect_sphere_queries(device_sphere_view point_spheres);

void inflate_sphere_query_radii(device_intersects_query_view queries, double factor);

template<typename query_view_type>
device_search_results do_search(device_point_view nodes, query_view_type queries);

}
}
}


