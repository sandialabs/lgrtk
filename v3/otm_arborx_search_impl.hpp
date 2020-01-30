#pragma once

#include <Kokkos_View.hpp>

namespace lgr {
class state;
}

namespace ArborX {
class Box;
class Point;
template <typename GeomType> class Nearest;
}

namespace lgr {
namespace search {
namespace arborx {

using device_exec_space = Kokkos::DefaultExecutionSpace;
using device_point_view = Kokkos::View<ArborX::Point *, device_exec_space>;
using device_nearest_query_view = Kokkos::View<ArborX::Nearest<ArborX::Point> *, device_exec_space>;
using device_int_view = Kokkos::View<int *, device_exec_space>;
using device_search_results = Kokkos::pair<device_int_view, device_int_view>;

device_point_view create_arborx_nodes(const lgr::state &s);

device_point_view create_arborx_points(const lgr::state &s);

device_nearest_query_view make_nearest_node_queries(device_point_view points,
    const int num_nodes_to_find);

device_search_results do_search(device_point_view nodes,
    device_nearest_query_view queries);

}
}
}


