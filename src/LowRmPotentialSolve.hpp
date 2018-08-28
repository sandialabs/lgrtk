/*
//@HEADER
// ************************************************************************
//
//                        lgr v. 1.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  Glen A. Hansen (gahanse@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef LOW_RM_POTENTIAL_SOLVE_HPP
#define LOW_RM_POTENTIAL_SOLVE_HPP

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPDecl.hpp>

#include <CellTools.hpp>
#include <CrsLinearProblem.hpp>
#include <CrsMatrix.hpp>
#include <Fields.hpp>
#include <MeshIO.hpp>
#include <ParallelComm.hpp>

namespace lgr {

using DefaultLocalOrdinal = int;
using DefaultGlobalOrdinal = long;
using DefaultLayout = Kokkos::LayoutRight;

// This one Tpetra likes; will have to check whether this works with Magma Sparse and AmgX or if we need to do something to factor this out
using RowMapEntryType = int;

using CrsMatrixType = typename lgr::CrsMatrix<
    DefaultLocalOrdinal,
    RowMapEntryType>;

template <int SpatialDim>
class LowRmPotentialSolve {
  using DefaultFields = Fields<SpatialDim>;

 public:
  typedef Kokkos::View<DefaultLocalOrdinal *, DefaultLayout, MemSpace>
      LocalOrdinalVector;
  typedef Kokkos::View<Scalar *, DefaultLayout, MemSpace>
      ScalarVector;
  //    typedef  DefaultFields::elem_sym_tensor_type SymTensorField;

  // 2D view for multiple RHSes
  typedef Kokkos::View<Scalar **, DefaultLayout, MemSpace>
      ScalarMultiVector;
  typedef Kokkos::View<Scalar ***, DefaultLayout, MemSpace>
      CellWorkset;

  typedef CrsLinearProblem<
      DefaultLocalOrdinal>
      CrsLinearSolver;
  using ConductivityType = typename DefaultFields::
      array_type;  // scalar-valued for now (could generalize to tensor-valued in future)
 private:
  CrsMatrixType     _matrix;
  ScalarMultiVector _lhs, _rhs;
  int               _mSeries = 1;    // the "m" for m-fold series symmetry
  int               _mParallel = 1;  // the "m" for m-fold parallel symmetry

 public:
  // Conceptually private methods, declared public just because these have extended __device__ lambdas defined,
  // and putting those in private or protected methods is verboten (per nvcc)
  void assemble(
      const CellTools::PhysCellGradientView &physicalGradients1,
      const CellTools::PhysCellGradientView &physicalGradients2,
      const CellWorkset &                    cellWorkset,
      const CellTools::FusedJacobianDetView &cellJacobians);

  // ! private method that will create a default identity conductivity if it does not already exist
  // ! second argument is of length symm_ncomps(spaceDim) -- number of components in a SymTensor
  ScalarMultiVector getConductivity();

  void initializeCellWorkset(CellWorkset &cellWorkset);

  // ! Accumulate into the stiffness matrix and RHSes -- new version meant to eliminate nearly all temporary allocations on device
  void fusedAssemble();

 private:
  comm::Machine               _machine;
  Teuchos::RCP<DefaultFields> _meshFields;
  int                         _numConductors;
  int                         _spaceDim;

  int         _quadratureDegreeForForcing;
  std::string _forcingFunctionExpr = "";

  // BCs:
  LocalOrdinalVector _bcNodes;
  ScalarVector       _bcValues;

  ConductivityType _conductivity;

  bool haveForcingFunction();

  Scalar _K11 = 0.0;
  Scalar _totalJoulesAdded =
      0.0;  // cumulative over all calls to determineJouleHeating
 public:
  LowRmPotentialSolve(
      Teuchos::ParameterList const &paramList,
      Teuchos::RCP<DefaultFields>   meshFields,
      comm::Machine                 machine);

  // ! Allocate storage for the stiffness matrix, LHS(es) and RHS(es)
  void initialize();

  // ! Accumulate into the stiffness matrix and RHSes
  void assemble();

  // ! resets the mesh to the one specified in meshFields.  Also clears _bcNodes and _bcValues.
  void resetMesh(Teuchos::RCP<DefaultFields> meshFields);

  // ! sets boundary conditions for the specified nodes by evaluating the supplied expression at those nodes
  // ! (the actual imposition of BCs occurs during assemble()).  Calls the setBC(LOs,Reals) method (below).
  void setBC(
      const std::string &bcExpr,
      const Omega_h::LOs localNodeOrdinals,
      bool               addToExisting = false);

  // ! sets boundary conditions for the specified nodes to the specified values.
  // ! (the actual imposition of BCs occurs during assemble())
  void setBC(
      const Omega_h::LOs   localNodeOrdinals,
      const Omega_h::Reals values,
      bool                 addToExisting = false);

  void setForcingFunctionExpr(
      const std::string &forcingFunctionExpr, int degreeForQuadrature);

  void setConductivity(ConductivityType conductivity);

  // ! set the input and output ports for the lgr circuit element.
  // ! This calls setBC(localNodeOrdinals, values), below.
  // ! The assumption is an RLC circuit; the input port is on the resistor side, and the output is on the capacitor side.
  void setPorts(
      const MeshIO &meshIO,
      std::string & inputNodeSetName,
      std::string & outputNodeSetName);

  void setUseMFoldSymmetry(int mSeries, int mParallel);

  // ! compute and return the top-left entry of the lumped circuit model element matrix K.
  // ! If we ever have more than 2 ports, we'll need to expose the whole K matrix to support the circuit solve.
  Scalar getK11();

  // ! Compute the energy contribution from Joule heating, and add it elementwise to the provided view.
  // ! The V3 voltage is the voltage value at the lgr "input" port in the RLC circuit.
  // ! Also computes the meshwise integral that is used in getK11().
  void determineJouleHeating(
      ScalarVector cellInternalEnergy,
      ScalarVector cellJouleEnergy,
      double       V3,
      double       dt,
      bool         warnIfCellInternalEnergyIsEmpty = true);

  // ! Returns the stiffness matrix
  CrsMatrixType getMatrix();

  // ! returns the solution vector
  ScalarMultiVector getLHS();

  // ! returns the RHS vector
  ScalarMultiVector getRHS();

  // ! convenience function that returns a constant diagonal conductivity tensor with value on the diagonal = constantValue
  ConductivityType getConstantConductivity(Scalar constantValue);

  // ! get a default solver for the problem.  Should be called after first call to assemble().  (After the first call, we should have some mechanism for letting the solver object know that the matrix/vector entries have changed, without recreating everything from scratch.  But as of this writing, 8/31/17, we don't have that yet.)
  Teuchos::RCP<CrsLinearSolver> getDefaultSolver(double tol, int maxIters);

  Scalar getTotalJoulesAdded();
};
}  // namespace lgr

#endif
