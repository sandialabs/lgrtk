/*
 * AlgebraicRocketTest.cpp
 *
 *  Created on: Oct 25, 2018
 *      Author: drnoble
 */

#include "plato/HamiltonJacobi.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include "LGRTestHelpers.hpp"

typename Omega_h::Mesh
build_2D_box(const Plato::Scalar Lx, const Plato::Scalar Ly, const size_t nx,
    const size_t ny)
{
  auto libOmegaH = lgr::getLibraryOmegaH();
  return Omega_h::build_box(libOmegaH->world(), OMEGA_H_SIMPLEX, Lx, Ly, 0, nx,
      ny, 0);
}

typename Omega_h::Mesh
build_3D_box(const Plato::Scalar Lx, const Plato::Scalar Ly,
    const Plato::Scalar Lz, const size_t nx, const size_t ny, const size_t nz)
{
  auto libOmegaH = lgr::getLibraryOmegaH();
  return Omega_h::build_box(libOmegaH->world(), OMEGA_H_SIMPLEX, Lx, Ly, Lz, nx,
      ny, nz);
}

struct PoorSignedDistanceForUnitCircle
{
  Plato::Scalar
  operator()(Plato::Scalar x, Plato::Scalar y, Plato::Scalar z) const
  {
    return ((x - 1.) * (x - 1.) + (y - 1.) * (y - 1.) + 0.1)
        * (sqrt(x * x + y * y) - 1.);
  }
};

struct PoorSignedDistanceForUnitSphere
{
  Plato::Scalar
  operator()(Plato::Scalar x, Plato::Scalar y, Plato::Scalar z) const
  {
    return ((x - 1.) * (x - 1.) + (y - 1.) * (y - 1.) + (z - 1.) * (z - 1.)
        + 0.1) * (sqrt(x * x + y * y + z * z) - 1.);
  }
};

struct Flower2DApproximateSignedDistance
{
  Plato::Scalar
  operator()(Plato::Scalar x, Plato::Scalar y, Plato::Scalar z) const
  {
    const int Nlobes = 3;
    const Plato::Scalar xCenter = 1.0;
    const Plato::Scalar yCenter = 1.0;
    const Plato::Scalar Rmid = 0.5;
    const Plato::Scalar Rdelta = 0.15;
    const Plato::Scalar r = std::sqrt(
        (x - xCenter) * (x - xCenter) + (y - yCenter) * (y - yCenter));
    const Plato::Scalar theta = atan2(y - yCenter, x - xCenter);
    const Plato::Scalar rSurf = Rmid + Rdelta * sin(Nlobes * theta);
    return r - rSurf;
  }
};

template<int SpatialDim>
  Plato::Scalar
  compute_unit_radius_error_norm(Omega_h::Mesh & omega_h_mesh,
      ProblemFields<SpatialDim> & fields)
  {
    auto levelSet = fields.mLevelSet;
    const Omega_h::Reals coords = omega_h_mesh.coords();

    auto f = LAMBDA_EXPRESSION(int n, Plato::Scalar &norm)
    {
      const Plato::Scalar x = coords[n*SpatialDim+0];
      const Plato::Scalar y = coords[n*SpatialDim+1];
      const Plato::Scalar z = (SpatialDim > 2) ? coords[n*SpatialDim+2] : 0.0;
      Plato::Scalar exact = sqrt(x*x+y*y+z*z)-1.;
      norm += abs(levelSet(n,fields.currentState) - exact);
    };

    Plato::Scalar norm = 0.0;
    Kokkos::parallel_reduce(omega_h_mesh.nverts(), f, norm);
    return norm / omega_h_mesh.nverts();
  }

TEUCHOS_UNIT_TEST(HamiltonJacobi, 2DPoorInitialCondition_ComputingArrivalTimeProducesLowErrorEverywhere)
{
  static constexpr int SpatialDim = 2;

  auto omega_h_mesh = build_2D_box(2., 2., 64, 64);

  ProblemFields<SpatialDim> fields;
  declare_fields(omega_h_mesh, fields);

  PoorSignedDistanceForUnitCircle initialCondition;
  initialize_level_set(omega_h_mesh, fields, initialCondition);

  initialize_constant_speed(omega_h_mesh, fields, 1.0);

  const Plato::Scalar dx = mesh_minimum_length_scale<SpatialDim>(omega_h_mesh);
  const Plato::Scalar eps = 1.5 * dx; // Should have same units as level set
  const Plato::Scalar dtau = 0.2 * dx; // Reinitialization time step, based on unit advection speed used in reinitialization

  compute_arrival_time(omega_h_mesh, fields, eps, dtau);

  const Plato::Scalar errorNorm = compute_unit_radius_error_norm(omega_h_mesh,
      fields);
  std::cout << "Error norm " << errorNorm << std::endl;
  TEST_COMPARE(errorNorm, <, 0.0011);
}

TEUCHOS_UNIT_TEST(HamiltonJacobi, 3DPoorInitialCondition_ComputingArrivalTimeProducesLowErrorEverywhere)
{
  static constexpr int SpatialDim = 3;

  auto omega_h_mesh = build_3D_box(2., 2., 2., 32, 32, 32);

  ProblemFields<SpatialDim> fields;
  declare_fields(omega_h_mesh, fields);

  PoorSignedDistanceForUnitSphere initialCondition;
  initialize_level_set(omega_h_mesh, fields, initialCondition);

  initialize_constant_speed(omega_h_mesh, fields, 1.0);

  const Plato::Scalar dx = mesh_minimum_length_scale<SpatialDim>(omega_h_mesh);
  const Plato::Scalar eps = 1.5 * dx; // Should have same units as level set
  const Plato::Scalar dtau = 0.2 * dx; // Reinitialization time step, based on unit advection speed used in reinitialization

  compute_arrival_time(omega_h_mesh, fields, eps, dtau);

  const Plato::Scalar errorNorm = compute_unit_radius_error_norm(omega_h_mesh,
      fields);
  std::cout << "Error norm " << errorNorm << std::endl;
  TEST_COMPARE(errorNorm, <, 0.006);
}

TEUCHOS_UNIT_TEST(HamiltonJacobi, 3D_CylinderWithFlowerIC_ReinitializationThenEvolveRuns)
{
  static constexpr int SpatialDim = 3;

  auto omega_h_mesh = build_3D_box(2., 2., 2., 32, 32, 32);

  ProblemFields<SpatialDim> fields;
  declare_fields(omega_h_mesh, fields);

  Flower2DApproximateSignedDistance initialCondition;
  initialize_level_set(omega_h_mesh, fields, initialCondition);

  const Plato::Scalar maxSpeed = initialize_constant_speed(omega_h_mesh, fields,
      5.0);

  const Plato::Scalar Courant = 0.3;
  const Plato::Scalar Ttotal = 0.05;
  const Plato::Scalar dx = mesh_minimum_length_scale<SpatialDim>(omega_h_mesh);
  const unsigned Nt = (Ttotal + 0.5 * (Courant * dx / maxSpeed))
      / (Courant * dx / maxSpeed);
  const Plato::Scalar dt = Ttotal / Nt;
  const Plato::Scalar eps = 1.5 * dx;
  const Plato::Scalar dtau = 0.2 * dx; // Reinitialization time step, based on unit advection speed used in reinitialization

  Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer("MyMesh", &omega_h_mesh,
      SpatialDim);

  Plato::Scalar time = 0.0;

  reinitialize_level_set(omega_h_mesh, fields, time, eps, dtau);

  for (unsigned n = 0; n < Nt; ++n)
  {
    evolve_level_set(omega_h_mesh, fields, eps, dt);
    time += dt;
    write_mesh(tWriter, omega_h_mesh, fields, time);
    reinitialize_level_set(omega_h_mesh, fields, time, eps, dtau);

    std::cout << "Time, Level set vol = " << time << " "
        << level_set_volume(omega_h_mesh, fields, eps)
        << std::endl;
  }
}
