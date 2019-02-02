#ifndef HAMILTONJACOBI_HPP
#define HAMILTONJACOBI_HPP

#include "Kokkos_Atomic.hpp"

#include "Omega_h_build.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_mesh.hpp"

#include "ImplicitFunctors.hpp"
#include "plato/PlatoStaticsTypes.hpp"

#include "Fields.hpp"

template<int SpatialDim>
KOKKOS_INLINE_FUNCTION
Plato::Scalar unitize(Omega_h::Vector<SpatialDim> & v)
{
  Plato::Scalar mag = 0.;
  for (int dim=0; dim<SpatialDim; ++dim) mag += v[dim] * v[dim];
  mag = sqrt(mag);
  for (int dim=0; dim<SpatialDim; ++dim) v[dim] /= mag;
  return mag;
}

template<int SpatialDim>
KOKKOS_INLINE_FUNCTION
Plato::Scalar length(const Omega_h::Vector<SpatialDim> & v)
{
  Plato::Scalar sqrMag = 0.;
  for (int dim=0; dim<SpatialDim; ++dim) sqrMag += v[dim] * v[dim];
  return sqrt(sqrMag);
}

template<int SpatialDim>
KOKKOS_INLINE_FUNCTION
Plato::Scalar length_squared(const Omega_h::Vector<SpatialDim> & v)
{
  Plato::Scalar sqrMag = 0.;
  for (int dim=0; dim<SpatialDim; ++dim) sqrMag += v[dim] * v[dim];
  return sqrMag;
}

template<int SpatialDim>
KOKKOS_INLINE_FUNCTION
Plato::Scalar dot(const Omega_h::Vector<SpatialDim> & v1, const Omega_h::Vector<SpatialDim> & v2)
{
  Plato::Scalar result = 0.;
  for (int dim=0; dim<SpatialDim; ++dim) result += v1[dim] * v2[dim];
  return result;
}

template<int SpatialDim>
struct ProblemFields
{
  typename lgr::Fields<SpatialDim>::state_array_type levelSet;
  typename lgr::Fields<SpatialDim>::array_type RHS;
  typename lgr::Fields<SpatialDim>::array_type RHSNorm;
  typename lgr::Fields<SpatialDim>::array_type speed;
  int currentState = 0;
};

template<int SpatialDim>
void declare_fields(Omega_h::Mesh & omega_h_mesh, ProblemFields<SpatialDim> & fields)
{
  const int elem_count = omega_h_mesh.nelems();
  const int node_count = omega_h_mesh.nverts();

  typedef lgr::Fields<SpatialDim> Fields;
  fields.levelSet = typename Fields::state_array_type("nodal levelSet", node_count);
  fields.RHS = typename Fields::array_type("nodal RHS", node_count);
  fields.RHSNorm = typename Fields::array_type("nodal RHS norm", node_count);
  fields.speed = typename Fields::array_type("element speed", elem_count);
}

template<int SpatialDim>
Plato::Scalar initialize_constant_speed(Omega_h::Mesh & omega_h_mesh, ProblemFields<SpatialDim> & fields, const Plato::Scalar speed)
{
  auto elemSpeed = fields.speed;
  auto f = LAMBDA_EXPRESSION(int elem) {
    elemSpeed(elem) = speed;
  };
  Kokkos::parallel_for(omega_h_mesh.nelems(), f);
  return speed;
}

KOKKOS_INLINE_FUNCTION
Plato::Scalar Heaviside(const Plato::Scalar signedDist, const Plato::Scalar eps)
{
  if (signedDist < -eps) return 0.;
  else if (signedDist > eps) return 1.;

  constexpr Plato::Scalar pi = 3.1415926535897932385;
  return 0.5*(1+ signedDist/eps + sin(pi*signedDist/eps)/pi);
}

KOKKOS_INLINE_FUNCTION
Plato::Scalar Delta(const Plato::Scalar signedDist, const Plato::Scalar eps)
{
  if (signedDist < -eps || signedDist > eps) return 0.;

  constexpr Plato::Scalar pi = 3.1415926535897932385;
  return 0.5/eps * (1. + cos(pi*signedDist/eps));
}

template<int SpatialDim, class Lambda>
Plato::Scalar level_set_integral(Omega_h::Mesh & omega_h_mesh, ProblemFields<SpatialDim> & fields, double eps, const Lambda & levelSetFn, double levelSet)
{
  constexpr int nodesPerElem = SpatialDim + 1;
  Plato::NodeCoordinate<SpatialDim> nodeCoordinate(&omega_h_mesh);
  Plato::ComputeVolume<SpatialDim> computeElemVolume(nodeCoordinate);
  auto levelSetField = fields.levelSet;
  auto elems2Verts = omega_h_mesh.ask_elem_verts();

  auto computeIntegral = LAMBDA_EXPRESSION(int elem, Plato::Scalar & integral) {
    Plato::Scalar elemVolume = computeElemVolume(elem);

    Plato::Scalar avgLS = 0.;
    for (int n=0; n<nodesPerElem; ++n)
    {
      avgLS += levelSetField(elems2Verts[elem * nodesPerElem + n],fields.currentState)/nodesPerElem;
    }

    integral += elemVolume * levelSetFn(avgLS - levelSet, eps);
  };
  Plato::Scalar integral = 0.;
  Kokkos::parallel_reduce(omega_h_mesh.nelems(), computeIntegral, integral);
  return integral;
}

template<int SpatialDim>
Plato::Scalar level_set_volume(Omega_h::Mesh & omega_h_mesh, ProblemFields<SpatialDim> & fields, double eps, double levelSet = 0.)
{
  auto fn = LAMBDA_EXPRESSION(const Plato::Scalar signedDist, const Plato::Scalar eps) {
    return Heaviside(signedDist, eps);
  };
  return level_set_integral(omega_h_mesh, fields, eps, fn, levelSet);
}

template<int SpatialDim>
Plato::Scalar level_set_area(Omega_h::Mesh & omega_h_mesh, ProblemFields<SpatialDim> & fields, double eps, double levelSet = 0.)
{
  auto fn = LAMBDA_EXPRESSION(const Plato::Scalar signedDist, const Plato::Scalar eps) {
    return Delta(signedDist, eps);
  };
  return level_set_integral(omega_h_mesh, fields, eps, fn, levelSet);
}

template<int SpatialDim>
Plato::Scalar mesh_minimum_length_scale(Omega_h::Mesh & omega_h_mesh)
{
  constexpr int nodesPerCell = SpatialDim + 1;
  Plato::NodeCoordinate<SpatialDim> nodeCoordinate(&omega_h_mesh);
  Plato::ComputeGradient<SpatialDim> computeGradient(nodeCoordinate);

  auto findMaxGradSqrMagnitude = LAMBDA_EXPRESSION(int e, Plato::Scalar & maxGradSqrMag) {
    Plato::Scalar elemVolume;
    Omega_h::Vector<SpatialDim> gradients[nodesPerCell];
    computeGradient(e, gradients, elemVolume);

    for (unsigned n=0; n<nodesPerCell; ++n)
    {
      const Plato::Scalar gradSqrMag = length_squared(gradients[n]);
      maxGradSqrMag = Omega_h::max2(maxGradSqrMag, gradSqrMag);
    }
  };

  Plato::Scalar maxGradSqrMag ;
  Kokkos::Max<Plato::Scalar> xreducer(maxGradSqrMag);
  Kokkos::parallel_reduce(omega_h_mesh.nelems(), findMaxGradSqrMagnitude, xreducer);

  return 1./sqrt(maxGradSqrMag);
}

template<typename TYPE, int SIZE>
struct SumArray
{
  KOKKOS_INLINE_FUNCTION
  SumArray() { for (int i=0; i<SIZE; ++i) data[i] = 0; }

  KOKKOS_INLINE_FUNCTION
  TYPE & operator[](int i) { return data[i]; }

  KOKKOS_INLINE_FUNCTION
  volatile SumArray<TYPE,SIZE> & operator+=(const volatile SumArray<TYPE,SIZE> & RHS) volatile
  {
    for (int i=0; i<SIZE; ++i) data[i] += RHS.data[i];
    return *this;
  }

  TYPE data[SIZE];
};

template<int SpatialDim>
bool domain_contains_interface(Omega_h::Mesh & omega_h_mesh, ProblemFields<SpatialDim> & fields)
{
  auto levelSet = fields.levelSet;

  auto f = LAMBDA_EXPRESSION(int n, SumArray<size_t,2> & numPosNeg) {
    if (levelSet(n,fields.currentState) < 0) ++numPosNeg[0];
    if (levelSet(n,fields.currentState) > 0) ++numPosNeg[1];
  };

  SumArray<size_t,2> numPosNeg;
  Kokkos::parallel_reduce(omega_h_mesh.nverts(), f, numPosNeg);

  return numPosNeg[0] > 0 && numPosNeg[1] > 0;
}

template<int SpatialDim, class Lambda>
void initialize_level_set(Omega_h::Mesh & omega_h_mesh,
    ProblemFields<SpatialDim> & fields,
    const Lambda & level_set_function,
    const double multiplier = 1.0)
{
  auto levelSet = fields.levelSet;
  const Omega_h::Reals coords = omega_h_mesh.coords();
  auto f = LAMBDA_EXPRESSION(int n) {
    const Plato::Scalar x = coords[n*SpatialDim+0];
    const Plato::Scalar y = coords[n*SpatialDim+1];
    const Plato::Scalar z = (SpatialDim > 2) ? coords[n*SpatialDim+2] : 0.0;
    levelSet(n, fields.currentState) = level_set_function(x,y,z) * multiplier;
  };
  Kokkos::parallel_for(omega_h_mesh.nverts(), f);
}

template<int SpatialDim>
void offset_level_set(Omega_h::Mesh & omega_h_mesh,
    ProblemFields<SpatialDim> & fields,
    const double offset)
{
  auto levelSet = fields.levelSet;
  auto f = LAMBDA_EXPRESSION(int n) {
    levelSet(n, fields.currentState) += offset;
  };
  Kokkos::parallel_for(omega_h_mesh.nverts(), f);
}

template<int SpatialDim>
class AssembleElementSpeedHamiltonian
{
  // Implementation of Barth-Sethian positive coefficient scheme for element speed fields.
private:
  const Omega_h::LOs mElems2Verts;
  const Plato::NodeCoordinate<SpatialDim> mNodeCoordinate;
  const Plato::ComputeGradient<SpatialDim> mComputeGradient;
  const ProblemFields<SpatialDim> mFields;
  const double mEps;
public:
  AssembleElementSpeedHamiltonian(Omega_h::Mesh & omega_h_mesh, ProblemFields<SpatialDim> & fields, const double eps) :
    mElems2Verts(omega_h_mesh.ask_elem_verts()),
    mNodeCoordinate(&omega_h_mesh),
    mComputeGradient(mNodeCoordinate),
    mFields(fields),
    mEps(eps) {}

  DEVICE_TYPE inline
  void operator()(int elem) const
  {
    constexpr int nodesPerElem = SpatialDim + 1;
    Plato::Scalar elemVolume;
    Omega_h::Vector<SpatialDim> gradients[nodesPerElem];
    mComputeGradient(elem, gradients, elemVolume);

    Omega_h::Vector<SpatialDim> normalDir;
    for (int dim=0; dim<SpatialDim; ++dim) normalDir[dim] = 0.;
    for (int n=0; n<nodesPerElem; ++n)
    {
      const Plato::Scalar LSOld = mFields.levelSet(mElems2Verts[elem * nodesPerElem + n], mFields.currentState);
      for (int dim=0; dim<SpatialDim; ++dim) normalDir[dim] += gradients[n][dim] * LSOld;
    }
    unitize(normalDir);

    const Plato::Scalar elementSpeed = mFields.speed[elem];
    Omega_h::Vector<nodesPerElem> volHamiltonianCoeffs; // K_i in Barth-Sethian
    double sumNegCoeffs = 0.;
    double sumPosCoeffs = 0.;
    double sumNegContrib = 0.;
    double volHamiltonian = 0.;
    for (int n=0; n<nodesPerElem; ++n)
    {
      auto node = mElems2Verts[elem * nodesPerElem + n];
      const Plato::Scalar LSOld = mFields.levelSet(node, mFields.currentState);
      volHamiltonianCoeffs[n] = elemVolume*elementSpeed * dot(normalDir, gradients[n]);
      volHamiltonian += volHamiltonianCoeffs[n] * LSOld;

      if (volHamiltonianCoeffs[n] < 0.)
      {
        sumNegCoeffs += volHamiltonianCoeffs[n];
        sumNegContrib += volHamiltonianCoeffs[n] * LSOld;
      }
      else
      {
        sumPosCoeffs += volHamiltonianCoeffs[n];
      }
    }

    Omega_h::Vector<nodesPerElem> alpha; // delta phi_i in Barth-Sethian
    double sumPosAlpha = 0.;
    for (int n=0; n<nodesPerElem; ++n)
    {
      auto node = mElems2Verts[elem * nodesPerElem + n];
      const Plato::Scalar LSOld = mFields.levelSet(node, mFields.currentState);
      if (volHamiltonianCoeffs[n] > 0.)
      {
        alpha[n] = volHamiltonianCoeffs[n]/sumPosCoeffs*(sumNegContrib-sumNegCoeffs*LSOld)/volHamiltonian;
        if (alpha[n] > 0.) sumPosAlpha += alpha[n];
      }
    }

    for (int n=0; n<nodesPerElem; ++n)
    {
      auto node = mElems2Verts[elem * nodesPerElem + n];
      const Plato::Scalar LSOld = mFields.levelSet(node, mFields.currentState);

      if (alpha[n] > 0.)
      {
        const Plato::Scalar wt = alpha[n]/sumPosAlpha;
        Kokkos::atomic_add(&mFields.RHS(node), wt * volHamiltonian);
        Kokkos::atomic_add(&mFields.RHSNorm(node), wt * elemVolume);
      }
    }
  }
};

template<int SpatialDim>
class AssembleNodalSpeedHamiltonian
{
  // Uses nodal speed to assemble nodal contributions for Hamiltonian.
  // Note that while the form supports any kind of nodal speed, this is currently specialized to
  // support Eikonal-type nodal speeds in get_nodal_speed().
private:
  const Omega_h::LOs mElems2Verts;
  const Plato::NodeCoordinate<SpatialDim> mNodeCoordinate;
  const Plato::ComputeGradient<SpatialDim> mComputeGradient;
  const ProblemFields<SpatialDim> mFields;
  const double mEps;
  const bool mComputeTimeOfArrival;

  DEVICE_TYPE inline
  Plato::Scalar get_nodal_speed(const ProblemFields<SpatialDim> & fields, int elem, int node, double eps) const
  {
    // Note that this could be generalized to support some other type of nodal speed.
    // For now just implement Eikonal-type nodal speed (either redistancing or time-of-arrival).
    const Plato::Scalar LSOld = fields.levelSet(node, fields.currentState);
    const Plato::Scalar sign = LSOld/sqrt(LSOld*LSOld + eps*eps);
    return (mComputeTimeOfArrival) ? sign*fields.speed[elem] : sign;
  }
public:
  AssembleNodalSpeedHamiltonian(Omega_h::Mesh & omega_h_mesh,
      ProblemFields<SpatialDim> & fields,
      const double eps,
      const bool computeTimeOfArrival) :
        mElems2Verts(omega_h_mesh.ask_elem_verts()),
        mNodeCoordinate(&omega_h_mesh),
        mComputeGradient(mNodeCoordinate),
        mFields(fields),
        mEps(eps),
        mComputeTimeOfArrival(computeTimeOfArrival) {}

  DEVICE_TYPE inline
  void operator()(int elem) const
  {
    constexpr int nodesPerElem = SpatialDim + 1;
    Plato::Scalar elemVolume;
    Omega_h::Vector<SpatialDim> gradients[nodesPerElem];
    mComputeGradient(elem, gradients, elemVolume);

    Omega_h::Vector<SpatialDim> normalDir;
    for (int dim=0; dim<SpatialDim; ++dim) normalDir[dim] = 0.;
    for (int n=0; n<nodesPerElem; ++n)
    {
      const Plato::Scalar LSOld = mFields.levelSet(mElems2Verts[elem * nodesPerElem + n], mFields.currentState);
      for (int dim=0; dim<SpatialDim; ++dim) normalDir[dim] += gradients[n][dim] * LSOld;
    }
    unitize(normalDir);

    for (int n=0; n<nodesPerElem; ++n)
    {
      auto node = mElems2Verts[elem * nodesPerElem + n];

      const Plato::Scalar nodalSpeed = get_nodal_speed(mFields, elem, node, mEps);
      const Plato::Scalar volHamiltonianCoeffs_n = elemVolume*nodalSpeed * dot(normalDir, gradients[n]);

      if (volHamiltonianCoeffs_n > 0.)
      {
        const Plato::Scalar LSOld = mFields.levelSet(node, mFields.currentState);
        Plato::Scalar sumNegCoeffs = 0.;
        Plato::Scalar sumPosCoeffs = 0.;
        Plato::Scalar sumNegContrib = 0.;
        Plato::Scalar volHamiltonian = 0.;
        for (int j=0; j<nodesPerElem; ++j)
        {
          const Plato::Scalar LSOldj = mFields.levelSet(mElems2Verts[elem * nodesPerElem + j], mFields.currentState);
          const Plato::Scalar volHamiltonianCoeffs_j = elemVolume*nodalSpeed * dot(normalDir, gradients[j]);

          volHamiltonian += volHamiltonianCoeffs_j * LSOldj;

          if (volHamiltonianCoeffs_j < 0.)
          {
            sumNegCoeffs += volHamiltonianCoeffs_j;
            sumNegContrib += volHamiltonianCoeffs_j * LSOldj;
          }
          else
          {
            sumPosCoeffs += volHamiltonianCoeffs_j;
          }
        }
        const Plato::Scalar wt = volHamiltonianCoeffs_n/sumPosCoeffs*(sumNegContrib-sumNegCoeffs*LSOld)/volHamiltonian;
        if (wt > 0.)
        {
          Kokkos::atomic_add(&mFields.RHS(node), wt * volHamiltonian);
          Kokkos::atomic_add(&mFields.RHSNorm(node), wt * elemVolume);
        }
      }
    }
  }
};

template<int SpatialDim>
void zero_RHS(Omega_h::Mesh & omega_h_mesh, ProblemFields<SpatialDim> & fields)
{
  auto RHS = fields.RHS;
  auto zeroRHS = LAMBDA_EXPRESSION(int n) { RHS(n) = 0.; };
  Kokkos::parallel_for(omega_h_mesh.nverts(), zeroRHS);

  auto RHSNorm = fields.RHSNorm;
  auto zeroRHSNorm = LAMBDA_EXPRESSION(int n) { RHSNorm(n) = 0.; };
  Kokkos::parallel_for(omega_h_mesh.nverts(), zeroRHSNorm);
}

template<int SpatialDim>
Plato::Scalar assemble_and_update_Eikonal(
    Omega_h::Mesh & omega_h_mesh,
    ProblemFields<SpatialDim> & fields,
    const Plato::Scalar eps,
    const Plato::Scalar dt,
    const bool computeArrivalTime)
{
  // This is a hybrid between the Barth-Sethian positive coefficient scheme and the Morgan-Waltz scheme for reinitialization.
  // Uses element based speed and nodal sign function to assemble nodal contributions for Hamiltonian (AssembleEikonalHamiltonian).
  // The assembled nodal Hamiltonian is then used with nodal source term to explicitly update signed distance
  // (or arrival time for non-unit speed).
  // Unlike the elemental algorithm developed by Barth-Sethian, this algorithm converges to the exact solution
  // for the "Distance Function Test" described in Morgan-Waltz.  Unlike the Morgan-Waltz algorithm, this
  // form converges much faster and is tolerant of meshes with obtuse angles because it uses the positive coefficient
  // form in Barth-Sethian.

  zero_RHS(omega_h_mesh, fields);

  AssembleNodalSpeedHamiltonian<SpatialDim> assemble(omega_h_mesh, fields, eps, computeArrivalTime);
  Kokkos::parallel_for(omega_h_mesh.nelems(), assemble);

  auto levelSet = fields.levelSet;
  const int currentState = fields.currentState;
  const int nextState = (currentState+1)%2;
  auto updateLevelSetAndReduceResidual = LAMBDA_EXPRESSION(int n, SumArray<Plato::Scalar,2> & residAndCount) {
    const Plato::Scalar Hamiltonian = (fields.RHSNorm(n) > 0.) ? (fields.RHS(n)/fields.RHSNorm(n)) : 0.;
    const Plato::Scalar sign = levelSet(n,currentState)/sqrt(levelSet(n,currentState)*levelSet(n,currentState) + eps*eps);

    levelSet(n,nextState) = levelSet(n,currentState) - dt * (Hamiltonian - sign);
    if (computeArrivalTime || abs(levelSet(n,nextState)) < eps)
    {
      residAndCount[0] += (Hamiltonian - sign)*(Hamiltonian - sign);
      residAndCount[1] += 1.0;
    }
  };

  SumArray<Plato::Scalar,2> residAndCount;
  Kokkos::parallel_reduce(omega_h_mesh.nverts(), updateLevelSetAndReduceResidual, residAndCount);

  fields.currentState = nextState;

  return std::sqrt(residAndCount[0]/residAndCount[1]);
}

template<int SpatialDim>
void evolve_level_set(
    Omega_h::Mesh & omega_h_mesh,
    ProblemFields<SpatialDim> & fields,
    const Plato::Scalar eps,
    const Plato::Scalar dt)
{
  // This is a uses the Barth-Sethian positive coefficient scheme for advection.

  zero_RHS(omega_h_mesh, fields);

  AssembleElementSpeedHamiltonian<SpatialDim> assemble(omega_h_mesh, fields, eps);
  Kokkos::parallel_for(omega_h_mesh.nelems(), assemble);

  const int currentState = fields.currentState;
  const int nextState = (currentState+1)%2;
  auto levelSet = fields.levelSet;
  auto updateLevelSet = LAMBDA_EXPRESSION(int n) {
    const Plato::Scalar Hamiltonian = (fields.RHSNorm(n) > 0.) ? (fields.RHS(n)/fields.RHSNorm(n)) : 0.;
    levelSet(n,nextState) = levelSet(n,currentState) - dt * Hamiltonian;
  };
  Kokkos::parallel_for(omega_h_mesh.nverts(), updateLevelSet);

  fields.currentState = nextState;
}

template<int SpatialDim>
void write_mesh(Omega_h::vtk::Writer & tWriter, Omega_h::Mesh & omega_h_mesh, ProblemFields<SpatialDim> & fields, const Plato::Scalar time)
{
  Kokkos::View<Omega_h::Real*> view("into", lgr::Fields<SpatialDim>::getFromSA(fields.levelSet,0).size());
  Kokkos::deep_copy(view, lgr::Fields<SpatialDim>::getFromSA(fields.levelSet,fields.currentState));
  omega_h_mesh.add_tag(Omega_h::VERT, "LevelSet", 1, Omega_h::Reals(Omega_h::Write<Omega_h::Real>(view)));

  auto tTags = Omega_h::vtk::get_all_vtk_tags(&omega_h_mesh,SpatialDim);
  tWriter.write(static_cast<Omega_h::Real>(time), tTags);
}

template<int SpatialDim>
void compute_arrival_time (
    Omega_h::Mesh & omega_h_mesh,
    ProblemFields<SpatialDim> & fields,
    const Plato::Scalar eps,
    const Plato::Scalar dtau,
    const Plato::Scalar convergedTol = 0.01)
{
  if (!domain_contains_interface(omega_h_mesh, fields))
  {
    std::cout << "Error, input level set field does not contain zero level set for initializing arrival time calculation." << std::endl;
    return;
  }
  Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer("MyMesh", &omega_h_mesh, SpatialDim);

  bool converged = false;
  const int maxIters = 5000;
  const int printFreq = 100;
  for (int iter = 0; iter<maxIters; ++iter)
  {
    const Plato::Scalar averageNodalResidual =  assemble_and_update_Eikonal(omega_h_mesh, fields, eps, dtau, true);

    if ((iter+1) % printFreq == 0)
    {
      std::cout << "After " << iter+1 << " iterations in compute_arrival_time, the relative error is " << averageNodalResidual << std::endl;
      write_mesh(tWriter, omega_h_mesh, fields, iter*dtau);
    }

    if (averageNodalResidual < convergedTol)
    {
      converged = true;
      std::cout << "compute_arrival_time converged after " << iter+1 << " iterations  with relative error of " << averageNodalResidual << std::endl;
      break;
    }
  }
  if (!converged)
  {
    std::cout << "compute_arrival_time failed to converge after " << maxIters << " iterations." << std::endl;
  }
}

template<int SpatialDim>
void reinitialize_level_set (
    Omega_h::Mesh & omega_h_mesh,
    ProblemFields<SpatialDim> & fields,
    const Plato::Scalar time,
    const Plato::Scalar eps,
    const Plato::Scalar dtau,
    const Plato::Scalar convergedTol = 0.01)
{
  if (!domain_contains_interface(omega_h_mesh, fields))
  {
    return;
  }

  bool converged = false;
  const int maxIters = 5000;
  const int printFreq = 100;
  for (int iter = 0; iter<maxIters; ++iter)
  {
    const Plato::Scalar averageNodalResidual = assemble_and_update_Eikonal(omega_h_mesh, fields, eps, dtau, false);

    if ((iter+1) % printFreq == 0)
    {
      std::cout << "After " << iter+1 << " iterations in reinitialize_level_set, the relative error is " << averageNodalResidual << std::endl;
    }

    if (averageNodalResidual < convergedTol)
    {
      converged = true;
      std::cout << "At time " << time << ", reinitialize_level_set converged after " << iter+1 << " iterations  with relative error of " << averageNodalResidual << std::endl;
      break;
    }
  }
  if (!converged)
  {
    std::cout << "At time " << time << ", reinitialize_level_set failed to converge after " << maxIters << " iterations." << std::endl;
  }
}
#endif
