#pragma once

#include <Kokkos_View.hpp>

namespace lgr {
class state;
}

namespace ArborX {
class Point;
}

namespace lgr {
namespace search {
namespace impl {

using device_exec_space = Kokkos::DefaultExecutionSpace;
using device_point_view = Kokkos::View<ArborX::Point *, device_exec_space>;

device_point_view create_arborx_nodes(const lgr::state& s);

device_point_view create_arborx_points(const lgr::state& s);

}
}
}


