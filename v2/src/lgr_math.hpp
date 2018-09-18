#ifndef LGR_TENSORS_HPP
#define LGR_TENSORS_HPP

#include <Omega_h_matrix.hpp>
#include <Omega_h_lie.hpp>

namespace lgr {

using Omega_h::Vector;
using Omega_h::Matrix;
using Omega_h::identity_matrix;
using Omega_h::resize;
using Omega_h::square;
using Omega_h::zero_vector;
using Omega_h::transpose;
using Omega_h::diagonal;
using Omega_h::fill_vector;
using Omega_h::log_spd;
using Omega_h::sqrt_spd;
using Omega_h::trace;

}

#endif
