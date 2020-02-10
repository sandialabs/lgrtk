#pragma once

#include <hpc_symmetric3x3.hpp>
#include <hpc_matrix3x3.hpp>
#include <j2/hardening.hpp>

namespace lgr {
HPC_DEVICE void neo_Hookean_point(hpc::matrix3x3<double> const &F, double const K, double const G,
    hpc::symmetric3x3<double> &sigma, double &Keff, double& Geff, double& potential);

HPC_DEVICE void variational_J2_point(hpc::matrix3x3<double> const &F, j2::Properties props,
    hpc::symmetric3x3<double> &sigma, double &Keff, double& Geff, double& potential, hpc::matrix3x3<double> &Fp);
}

