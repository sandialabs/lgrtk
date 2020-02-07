#pragma once

#include <hpc_symmetric3x3.hpp>
#include <hpc_matrix3x3.hpp>

namespace lgr {
HPC_DEVICE void neo_Hookean_point(hpc::matrix3x3<double> const &F, double const K, double const G,
    hpc::symmetric3x3<double> &sigma, double &Keff, double& Geff, double& potential);
}

