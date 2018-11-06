#ifndef LGR_QUADRATIC_HPP
#define LGR_QUADRATIC_HPP

#include <Omega_h_adj.hpp>

namespace Omega_h {
class Mesh;
}

namespace lgr {

Omega_h::Few<Omega_h::LOs, 2> number_p2_nodes(Omega_h::Mesh& mesh);

Omega_h::LOs build_p2_elems2nodes(Omega_h::Mesh& mesh,
    Omega_h::Few<Omega_h::LOs, 2> nodes);

Omega_h::Adj build_p2_nodes2elems(Omega_h::Mesh& mesh,
    Omega_h::Few<Omega_h::LOs, 2> nodes);

Omega_h::Reals build_p2_node_coords(Omega_h::Mesh& mesh,
    Omega_h::Few<Omega_h::LOs, 2> nodes);

}

#endif
