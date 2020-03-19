#pragma once

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif

#include <Kokkos_Core.hpp>
#include <Kokkos_View.hpp>

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include <ArborX_Point.hpp>
#include <ArborX_Predicates.hpp>

namespace lgr {
class state;
}

namespace ArborX {
struct Box;
class Point;
struct Sphere;
template <typename GeomType> struct Nearest;
template <typename GeomType> struct Intersects;
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

void initialize();

void finalize();

device_point_view create_arborx_nodes(const lgr::state &s);

device_point_view create_arborx_points(const lgr::state &s);

device_sphere_view create_arborx_point_spheres(const lgr::state &s);

device_nearest_query_view make_nearest_node_queries(device_point_view points,
    const int num_nodes_to_find);

device_intersects_query_view make_intersect_sphere_queries(device_sphere_view point_spheres);

void inflate_sphere_query_radii(device_intersects_query_view queries, double factor);

template<typename query_view_type>
void do_search(device_point_view nodes, query_view_type queries, device_int_view &indices,
    device_int_view &offsets);

}
}
}


