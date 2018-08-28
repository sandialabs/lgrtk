#include <Omega_h_assoc.hpp>
#include <Omega_h_expr.hpp>
#include <Omega_h_matrix.hpp>

#include <LowRmPotentialSolve.hpp>

#include <Basis.hpp>
#include <CellTools.hpp>
#include <Cubature.hpp>
#include <ErrorHandling.hpp>

#ifdef HAVE_VIENNA_CL
#include "ViennaSparseLinearProblem.hpp"
#endif

#ifdef HAVE_AMGX
#include "AmgXSparseLinearProblem.hpp"
#endif

#include <cassert>

namespace lgr {

template <class T>
Omega_h::Write<T> getArray_Omega_h(std::string name, int entryCount) {
  return Omega_h::Write<T>(entryCount, name);
}

template <int SpatialDim>
LowRmPotentialSolve<SpatialDim>::LowRmPotentialSolve(
    Teuchos::ParameterList const &,
    Teuchos::RCP<DefaultFields>   meshFields,
    comm::Machine                 machine)
    : _machine(machine), _meshFields(meshFields) {
  _numConductors =
      0;  // TODO: parse paramList to see what caller says about the conductor count
  _spaceDim = _meshFields->femesh.omega_h_mesh->dim();

  // I *think* Kokkos won't like it if we resize() these arrays before we initialize them.
  // Since this can happen if there are no input/output ports and we have voltage BCs set,
  // we initialize them here.
  _bcNodes = LocalOrdinalVector("BC nodes", 0);
  _bcValues = ScalarVector("BC values", 0);
}

template <int spaceDim>
void multiplySymTensorByGradient(
    CellTools::PhysCellGradientView result,
    const typename LowRmPotentialSolve<spaceDim>::ScalarMultiVector
                                          symTensorView,
    const CellTools::PhysCellGradientView gradientView) {
  int                           numCells = gradientView.extent(0);
  const int                     nodesPerCell = spaceDim + 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<int>(0, numCells),
      LAMBDA_EXPRESSION(int cellOrdinal) {
        // get conductivity as an Omega_h matrix
        constexpr auto numSymComponents = Omega_h::symm_ncomps(spaceDim);
        using GradientVector = Omega_h::Vector<spaceDim>;
        using SymTensorVector = Omega_h::Vector<numSymComponents>;
        using SymMatrix = Omega_h::Matrix<spaceDim, spaceDim>;

        SymTensorVector symTensorVector;
        for (int comp = 0; comp < numSymComponents; comp++) {
          symTensorVector[comp] = symTensorView(cellOrdinal, comp);
        }

        SymMatrix matrix = vector2symm(symTensorVector);

        for (int nodeOrdinal = 0; nodeOrdinal < nodesPerCell; nodeOrdinal++) {
          GradientVector physicalGradient;
          for (int d = 0; d < spaceDim; d++) {
            physicalGradient[d] = gradientView(cellOrdinal, nodeOrdinal, d);
          }
          GradientVector resultVector = matrix * physicalGradient;
          for (int d = 0; d < spaceDim; d++) {
            result(cellOrdinal, nodeOrdinal, d) = resultVector[d];
          }
        }
      },
      "multiplySymTensorByGradient");
}

template <int spaceDim>
void multiplyConductivityByGradient(
    CellTools::PhysCellGradientView                                result,
    const typename LowRmPotentialSolve<spaceDim>::ConductivityType conductivity,
    const CellTools::PhysCellGradientView gradientView) {
  int                           numCells = gradientView.extent(0);
  const int                     nodesPerCell = spaceDim + 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<int>(0, numCells),
      LAMBDA_EXPRESSION(int cellOrdinal) {
        // get conductivity as an Omega_h matrix
        using GradientVector = Omega_h::Vector<spaceDim>;
        auto conductivityValue = conductivity(cellOrdinal);

        for (int nodeOrdinal = 0; nodeOrdinal < nodesPerCell; nodeOrdinal++) {
          GradientVector physicalGradient;
          for (int d = 0; d < spaceDim; d++) {
            physicalGradient[d] = gradientView(cellOrdinal, nodeOrdinal, d);
          }
          GradientVector resultVector = conductivityValue * physicalGradient;
          for (int d = 0; d < spaceDim; d++) {
            result(cellOrdinal, nodeOrdinal, d) = resultVector[d];
          }
        }
      },
      "multiplySymTensorByGradient");
}

template <int spaceDim>
void LowRmPotentialSolve<spaceDim>::assemble() {
  const int nodesPerCell = spaceDim + 1;
  int       numCells = _meshFields->femesh.nelems;

  TEUCHOS_TEST_FOR_EXCEPTION(
      _conductivity.extent(0) != _meshFields->femesh.nelems,
      std::invalid_argument,
      "_conductivity dimension 0 (" << _conductivity.extent(0)
                                    << ") must match nelems ("
                                    << _meshFields->femesh.nelems << ")");
  // the commented-out assertion below belongs to the tensor version:
  //  TEUCHOS_TEST_FOR_EXCEPTION(_conductivity.dimension_1() != symm_ncomps(spaceDim),     std::invalid_argument, "_conductivity dimension 1 must match symm_ncomps(spaceDim)");

  // clear any existing values
  Kokkos::deep_copy(_rhs, 0.0);
  Kokkos::deep_copy(_matrix.entries(), 0.0);
  _K11 = 0.0;

  bool useFusedAssemble = true;

  if (useFusedAssemble) {
    fusedAssemble();
  } else {
    //    cout << "\n\n**** regularAssemble ****\n\n";

    CellTools::FusedJacobianView<spaceDim> jacobian("jacobian", numCells);
    CellTools::FusedJacobianView<spaceDim> jacobianInv(
        "jacobian inverse", numCells);
    CellTools::FusedJacobianDetView jacobianDet("jacobian det.", numCells);

    static_assert(std::is_same<CellTools::PhysPointsView, CellWorkset>::value,
        "expected PhysPointsView and CellWorkset to be the same type!");
    CellTools::PhysPointsView cellWorkset(
        "cell workset", numCells, nodesPerCell, spaceDim);
    initializeCellWorkset(cellWorkset);

    CellTools::setFusedJacobian(jacobian, cellWorkset);
    CellTools::setFusedJacobianDet(jacobianDet, jacobian);
    CellTools::setFusedJacobianInv(jacobianInv, jacobian);

    // element vector view for conductivity * grad result:
    CellTools::PhysCellGradientView conductivityTimesGrad(
        "eps times basis->grad()", numCells, nodesPerCell, spaceDim);
    CellTools::PhysCellGradientView physicalGradients(
        "gradients", numCells, nodesPerCell, spaceDim);

    // start by loading the gradients:
    CellTools::getPhysicalGradients(physicalGradients, jacobianInv);

    //    { // DEBUGGING
    //      for (int d1=0; d1<spaceDim; d1++)
    //      {
    //        for (int d2=0; d2<spaceDim; d2++)
    //        {
    //          cout << "jacobian(0)[" << d1 << "]" << "[" << d2 << "] = " << jacobian(0)[d1][d2] << endl;
    //        }
    //      }
    //      for (int d1=0; d1<spaceDim; d1++)
    //      {
    //        for (int d2=0; d2<spaceDim; d2++)
    //        {
    //          cout << "jacobianInverse(0)[" << d1 << "]" << "[" << d2 << "] = " << jacobianInv(0)[d1][d2] << endl;
    //        }
    //      }
    //      cout << "jacobianDet(0) = " << jacobianDet(0) << endl;
    //
    //      for (int d1=0; d1<spaceDim; d1++)
    //      {
    //        for (int nodeOrdinal=0; nodeOrdinal<=spaceDim; nodeOrdinal++)
    //        {
    //          cout << "physicalGradients(0," << nodeOrdinal << "," << d1 << ") = " << physicalGradients(0, nodeOrdinal, d1) << endl;
    //        }
    //      }
    //    }

    multiplyConductivityByGradient<spaceDim>(
        conductivityTimesGrad, _conductivity, physicalGradients);
    assemble(
        conductivityTimesGrad, physicalGradients, cellWorkset, jacobianDet);
  }
}

template <int SpatialDim>
void LowRmPotentialSolve<SpatialDim>::assemble(
    const CellTools::PhysCellGradientView &physicalGradients1,
    const CellTools::PhysCellGradientView &physicalGradients2,
    const CellWorkset &                    cellWorkset,
    const CellTools::FusedJacobianDetView &jacobianDet) {
  using GradientVector = Omega_h::Vector<SpatialDim>;
  //  using GlobalOrdinal = DefaultGlobalOrdinal; // TODO: once we have MPI support, figure out how we will handle assembly differently (this will depend on using something other than our simple CrsMatrix, presumably -- probably a DistributedCrsMatrix that know something about how to communicate accumulated values to owners).
  const int nodesPerCell = SpatialDim + 1;

  auto mesh = _meshFields->femesh.omega_h_mesh;
  auto cells2nodes = mesh->ask_elem_verts();
  auto node_local2global = mesh->globals(0);

  int numCells = physicalGradients1.extent(0);

  /*
   How do I look up the entry ordinal, given a cell ordinal and two nodes in the cell?
   
   Local row ordinal can be computed as follows:
   DefaultLocalOrdinal iLocalOrdinal = cells2nodes[cellOrdinal + nodesPerCell * iNode];
   
   Global row ordinal is this:
   GlobalOrdinal iGlobalOrdinal = node_local2global[ iLocalOrdinal ];
   
   Presumably, it's the local row ordinal I want anyway, at least for owned nodes.  (It's not yet clear what we'll
   do with ghosts vis-a-vis CrsMatrix -- at present it is on-node only, with no MPI support.)
   
   One way to go is to simply search for the column index inside the CrsMatrix columnIndices, within the row entries.
   I've now asked Dan about this, and he suggests that this might not be too bad, though there is something more sophisticated
   we could try later -- using edge information from Omega_h.
   
   Note that some solvers (ViennaCL is one) do require that the matrix be strictly SPD: no asymmetries from BCs, and no symmetric negative definite matrices.
   
   This has led us to two revisions: first, we do impose symmetry during BC imposition; second, we reverse the signs on the matrix and the RHS to make the matrix SPD.
   
   On that second point: we understand the forcing function f, if provided, to be equal to div ( sigma * grad phi ); when we integrate by parts we get
      (- grad phi, sigma grad phi) = (f, phi).
   To get an SPD matrix, we instead compute
      (grad phi, sigma grad phi),
   which means that the RHS becomes (-f, phi).
   
   */

  auto rowMap = _matrix.rowMap();
  auto columnIndices = _matrix.columnIndices();
  auto entryOrdinalLookup =
      LAMBDA_EXPRESSION(int cellOrdinal, int iNode, int jNode) {
        DefaultLocalOrdinal iLocalOrdinal = cells2nodes[cellOrdinal * nodesPerCell + iNode];
        DefaultLocalOrdinal jLocalOrdinal = cells2nodes[cellOrdinal * nodesPerCell + jNode];
        RowMapEntryType rowStart = rowMap(iLocalOrdinal);
        RowMapEntryType rowEnd = rowMap(iLocalOrdinal + 1);
        for (RowMapEntryType entryOrdinal = rowStart; entryOrdinal < rowEnd; entryOrdinal++) {
          if (columnIndices(entryOrdinal) == jLocalOrdinal) {
            return entryOrdinal;
          }
        }
        //    cout << "Entry for (" << cellOrdinal << ", " << iNode << ", " << jNode << ") not found.  ";
        //    cout << "iLocalOrdinal " << iLocalOrdinal << "; ";
        //    cout << "jLocalOrdinal " << jLocalOrdinal << "; ";
        //    cout << "rowStart "      << rowStart      << "; ";
        //    cout << "rowEnd "        << rowEnd        << "; " << endl;
        return RowMapEntryType(-1);
      };

  ScalarVector matrixEntries = _matrix.entries();
  Scalar quadratureWeight = 1.0;  // for a 1-point quadrature rule for simplices
  for (int d = 2; d <= SpatialDim; d++) {
    quadratureWeight /= Scalar(d);
  }
  Kokkos::parallel_for(
      Kokkos::RangePolicy<int>(0, numCells),
      LAMBDA_EXPRESSION(int cellOrdinal) {
        auto   entriesLength = matrixEntries.size();
        Scalar cellVolume = fabs(jacobianDet(cellOrdinal)) * quadratureWeight;
        for (int iNode = 0; iNode < nodesPerCell; iNode++) {
          GradientVector iGradient;
          for (int d = 0; d < SpatialDim; d++) {
            iGradient[d] = physicalGradients1(cellOrdinal, iNode, d);
          }
          for (int jNode = 0; jNode < nodesPerCell; jNode++) {
            GradientVector jGradient;
            for (int d = 0; d < SpatialDim; d++) {
              jGradient[d] = physicalGradients2(cellOrdinal, jNode, d);
            }
            Scalar integral = (iGradient * jGradient) * cellVolume;

            auto entryOrdinal = entryOrdinalLookup(cellOrdinal, iNode, jNode);
            if (entryOrdinal < RowMapEntryType(entriesLength)) {
              Kokkos::atomic_add(&matrixEntries(entryOrdinal), integral);
            }
          }
        }
      },
      "grad-grad integration");

  //  {
  //    // DEBUGGING
  //    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
  //    {
  //      for (int iNode=0; iNode<nodesPerCell; iNode++)
  //      {
  //        for (int jNode=0; jNode<nodesPerCell; jNode++)
  //        {
  //          auto entryOrdinal = entryOrdinalLookup(cellOrdinal,iNode,jNode);
  //          cout << "K(" << cellOrdinal << "," << iNode << "," << jNode << ") = " << matrixEntries(entryOrdinal) << endl;
  //        }
  //      }
  //    }
  //  }

  // integrate the RHS
  // We do take advantage of the fact that HGRAD transform VALUE is an identity map; so we can just evaluate basis values once
  // at the reference quadrature points.
  // We do need to evaluate the forcing function at the physical quadrature points, though, if we have one
  // (if we don't, the RHS is 0 until we impose BCs, at least until we add other parameters like an externally applied electric
  //  field to the problem.)

  if (haveForcingFunction()) {
    int quadratureDegree =
        1 +
        _quadratureDegreeForForcing;  // degree of (linear) basis, plus nominal degree of forcing function

    int numPoints =
        Cubature::getNumCubaturePoints(SpatialDim, quadratureDegree);
    Basis basis(SpatialDim);
    int   numFields = basis.basisCardinality();

    Kokkos::View<Scalar **, Kokkos::LayoutRight, MemSpace> refCellQuadraturePoints(
        "ref quadrature points", numPoints, SpatialDim);
    Kokkos::View<Scalar **, Kokkos::LayoutRight, MemSpace> refCellBasisValues(
        "ref basis values", numFields, numPoints);
    Kokkos::View<Scalar ***, Kokkos::LayoutRight, MemSpace> quadraturePoints(
        "quadrature points", numCells, numPoints, SpatialDim);
    Kokkos::View<Scalar *, MemSpace> quadratureWeights(
        "quadrature weights", numPoints);

    Cubature::getCubature(
        SpatialDim, quadratureDegree, refCellQuadraturePoints,
        quadratureWeights);
    basis.getValues(refCellQuadraturePoints, refCellBasisValues);
    CellTools::mapToPhysicalFrame(
        quadraturePoints, refCellQuadraturePoints, cellWorkset);

    auto x_coords = getArray_Omega_h<double>(
        "forcing function x coords", numCells * numPoints);
    auto y_coords = getArray_Omega_h<double>(
        "forcing function y coords", numCells * numPoints);
    auto z_coords = getArray_Omega_h<double>(
        "forcing function z coords", numCells * numPoints);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<int>(0, numCells),
        LAMBDA_EXPRESSION(int cellOrdinal) {
          int entryOffset = cellOrdinal * numPoints;
          for (int ptOrdinal = 0; ptOrdinal < numPoints; ptOrdinal++) {
            if (SpatialDim > 0)
              x_coords[entryOffset + ptOrdinal] =
                  quadraturePoints(cellOrdinal, ptOrdinal, 0);
            if (SpatialDim > 1)
              y_coords[entryOffset + ptOrdinal] =
                  quadraturePoints(cellOrdinal, ptOrdinal, 1);
            if (SpatialDim > 2)
              z_coords[entryOffset + ptOrdinal] =
                  quadraturePoints(cellOrdinal, ptOrdinal, 2);
          }
        },
        "fill coords");

    Omega_h::ExprReader reader(numCells * numPoints, SpatialDim);
    if (SpatialDim > 0)
      reader.register_variable("x", Teuchos::any(Omega_h::Reals(x_coords)));
    if (SpatialDim > 1)
      reader.register_variable("y", Teuchos::any(Omega_h::Reals(y_coords)));
    if (SpatialDim > 2)
      reader.register_variable("z", Teuchos::any(Omega_h::Reals(z_coords)));

    Teuchos::any result;
    reader.read_string(result, _forcingFunctionExpr, "Low Rm forcing function");
    reader.repeat(result);
    auto fxnValues = Teuchos::any_cast<Omega_h::Reals>(result);

    auto rhs = _rhs;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<int>(0, numCells),
        LAMBDA_EXPRESSION(int cellOrdinal) {
          int entryOffset = cellOrdinal * numPoints;

          for (int ptOrdinal = 0; ptOrdinal < numPoints; ptOrdinal++) {
            Scalar fxnValue = fxnValues[entryOffset + ptOrdinal];

            Scalar weight =
                quadratureWeights(ptOrdinal) * fabs(jacobianDet(cellOrdinal));
            for (int fieldOrdinal = 0; fieldOrdinal < numFields;
                 fieldOrdinal++) {
              DefaultLocalOrdinal localOrdinal =
                  cells2nodes[cellOrdinal * nodesPerCell + fieldOrdinal];
              // TODO: revise the "0" below if/when we support multiple RHSes
              auto contribution = -weight * fxnValue *
                                  refCellBasisValues(fieldOrdinal, ptOrdinal);
              Kokkos::atomic_add(&rhs(0, localOrdinal), contribution);
            }
          }
        },
        "assemble RHS");
  }

  /*
   Because both MAGMA and ViennaCL apparently assume symmetry even though it's technically
   not required for CG (and they do so in a way that breaks the solve badly), we do make the
   effort here to maintain symmetry while imposing BCs.
   */
  int  numBCs = _bcNodes.size();
  auto rhs = _rhs;
  auto bcNodes = _bcNodes;
  auto bcValues = _bcValues;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<int>(0, numBCs),
      LAMBDA_EXPRESSION(int bcOrdinal) {
        DefaultLocalOrdinal nodeNumber = bcNodes[bcOrdinal];
        Scalar              value = bcValues[bcOrdinal];
        RowMapEntryType     rowStart = rowMap(nodeNumber);
        RowMapEntryType     rowEnd = rowMap(nodeNumber + 1);
        for (RowMapEntryType entryOrdinal = rowStart; entryOrdinal < rowEnd;
             entryOrdinal++) {
          DefaultLocalOrdinal column = columnIndices(entryOrdinal);
          if (column == nodeNumber)  // diagonal
          {
            matrixEntries(entryOrdinal) = 1.0;
          } else {
            // correct the rhs to account for the fact that we'll be zeroing out (col,row) as well
            // to maintain symmetry
            Kokkos::atomic_add(
                &rhs(0, column), -matrixEntries(entryOrdinal) * value);
            matrixEntries(entryOrdinal) = 0.0;
            RowMapEntryType colRowStart = rowMap(column);
            RowMapEntryType colRowEnd = rowMap(column + 1);
            for (RowMapEntryType colRowEntryOrdinal = colRowStart;
                 colRowEntryOrdinal < colRowEnd; colRowEntryOrdinal++) {
              DefaultLocalOrdinal colRowColumn =
                  columnIndices(colRowEntryOrdinal);
              if (colRowColumn == nodeNumber) {
                // this is the (col, row) entry -- clear it, too
                matrixEntries(colRowEntryOrdinal) = 0.0;
              }
            }
          }
        }
        rhs(0, nodeNumber) = value;
      },
      "BC imposition");
}

template <int spaceDim>
void LowRmPotentialSolve<spaceDim>::fusedAssemble() {
  constexpr int nodesPerCell = spaceDim + 1;

  auto mesh = _meshFields->femesh.omega_h_mesh;
  auto cells2nodes = mesh->ask_elem_verts();
  auto node_local2global = mesh->globals(0);

  int numCells = _meshFields->femesh.nelems;

  auto rowMap = _matrix.rowMap();
  auto columnIndices = _matrix.columnIndices();
  auto entryOrdinalLookup =
      LAMBDA_EXPRESSION(int cellOrdinal, int iNode, int jNode) {
    DefaultLocalOrdinal iLocalOrdinal =
        cells2nodes[cellOrdinal * nodesPerCell + iNode];
    DefaultLocalOrdinal jLocalOrdinal =
        cells2nodes[cellOrdinal * nodesPerCell + jNode];
    RowMapEntryType rowStart = rowMap(iLocalOrdinal);
    RowMapEntryType rowEnd = rowMap(iLocalOrdinal + 1);
    for (RowMapEntryType entryOrdinal = rowStart; entryOrdinal < rowEnd;
         entryOrdinal++) {
      if (columnIndices(entryOrdinal) == jLocalOrdinal) {
        return entryOrdinal;
      }
    }
    return RowMapEntryType(-1);
  };

  //  cout << "\n\n**** fusedAssemble ****\n\n";

  ScalarVector matrixEntries = _matrix.entries();
  Scalar quadratureWeight = 1.0;  // for a 1-point quadrature rule for simplices
  for (int d = 2; d <= spaceDim; d++) {
    quadratureWeight /= Scalar(d);
  }

  auto coords = mesh->coords();
  auto cellWorkset =
      LAMBDA_EXPRESSION(int cellOrdinal, int nodeOrdinal, int d) {
    DefaultLocalOrdinal vertexNumber =
        cells2nodes[cellOrdinal * nodesPerCell + nodeOrdinal];
    Scalar coord = coords[vertexNumber * spaceDim + d];
    //    cout << "cellWorkset(" << cellOrdinal << "," << nodeOrdinal << "," << d << ") = " << coord << endl;
    return coord;
  };

  auto conductivity = _conductivity;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<int>(0, numCells),
      LAMBDA_EXPRESSION(int cellOrdinal) {
        auto                       entriesLength = matrixEntries.size();
        Omega_h::Matrix<spaceDim, spaceDim> jacobian, jacobianInverse;

        // compute jacobian/Det/inverse for cell:
        for (int d1 = 0; d1 < spaceDim; d1++) {
          for (int d2 = 0; d2 < spaceDim; d2++) {
            jacobian[d1][d2] = cellWorkset(cellOrdinal, d2, d1) -
                               cellWorkset(cellOrdinal, spaceDim, d1);
          }
        }
        Scalar jacobianDet = Omega_h::determinant(jacobian);
        jacobianInverse = Omega_h::invert(jacobian);

        Scalar cellVolume = fabs(jacobianDet) * quadratureWeight;

        Omega_h::Vector<spaceDim> gradients[nodesPerCell];
        {
          // ref gradients in 3D are:
          //    field 0 = ( 1 ,0, 0)
          //    field 1 = ( 0, 1, 0)
          //    field 2 = ( 0, 0, 1)
          //    field 3 = (-1,-1,-1)

          // Therefore, when we multiply by the transpose jacobian inverse (which is what we do to compute physical gradients),
          // we have the following values:
          //    field 0 = row 0 of jacobianInv
          //    field 1 = row 1 of jacobianInv
          //    field 2 = row 2 of jacobianInv
          //    field 3 = negative sum of the three rows

          for (int d = 0; d < spaceDim; d++) {
            gradients[spaceDim][d] = 0.0;
          }

          for (int nodeOrdinal = 0; nodeOrdinal < spaceDim;
               nodeOrdinal++)  // "d1" for jacobian
          {
            for (int d = 0; d < spaceDim; d++)  // "d2" for jacobian
            {
              gradients[nodeOrdinal][d] = jacobianInverse[nodeOrdinal][d];
              gradients[spaceDim][d] -= jacobianInverse[nodeOrdinal][d];
            }
          }
        }

        // tensor version commented out:
        //    // get conductivity as an Omega_h matrix
        //    const int numSymComponents = symm_ncomps(spaceDim);
        //    using SymTensorVector = Vector<symm_ncomps(spaceDim)>;
        //    using SymMatrix = Matrix<spaceDim, spaceDim>;
        //
        //    SymTensorVector symTensorVector;
        //    for (int comp=0; comp<numSymComponents; comp++)
        //    {
        //      symTensorVector[comp] = conductivity(cellOrdinal,comp);
        //    }
        //
        //    SymMatrix cellConductivity = vector2symm(symTensorVector);
        auto cellConductivity = conductivity(cellOrdinal);

        for (int iNode = 0; iNode < nodesPerCell; iNode++) {
          for (int jNode = 0; jNode < nodesPerCell; jNode++) {
            Scalar integral =
                (gradients[iNode] * (cellConductivity * gradients[jNode])) *
                cellVolume;

            //        cout << "integral = " << integral << endl;

            RowMapEntryType entryOrdinal = entryOrdinalLookup(cellOrdinal, iNode, jNode);
            if (entryOrdinal < RowMapEntryType(entriesLength)) {
              Kokkos::atomic_add(&matrixEntries(entryOrdinal), integral);
            }
          }
        }
      },
      "grad-grad integration");

  // integrate the RHS
  // We do take advantage of the fact that HGRAD transform VALUE is an identity map; so we can just evaluate basis values once
  // at the reference quadrature points.
  // We do need to evaluate the forcing function at the physical quadrature points, though, if we have one
  // (if we don't, the RHS is 0 until we impose BCs, at least until we add other parameters like an externally applied electric
  //  field to the problem.)

  if (haveForcingFunction()) {
    int quadratureDegree =
        1 +
        _quadratureDegreeForForcing;  // degree of (linear) basis, plus nominal degree of forcing function

    int numPoints = Cubature::getNumCubaturePoints(spaceDim, quadratureDegree);
    Basis basis(spaceDim);
    int   numFields = basis.basisCardinality();

    Kokkos::View<Scalar **, Kokkos::LayoutRight, MemSpace> refCellQuadraturePoints(
        "ref quadrature points", numPoints, spaceDim);
    Kokkos::View<Scalar **, Kokkos::LayoutRight, MemSpace> refCellBasisValues(
        "ref basis values", numFields, numPoints);
    Kokkos::View<Scalar ***, Kokkos::LayoutRight, MemSpace> quadraturePoints(
        "quadrature points", numCells, numPoints, spaceDim);
    Kokkos::View<Scalar *, MemSpace> quadratureWeights(
        "quadrature weights", numPoints);

    Cubature::getCubature(
        spaceDim, quadratureDegree, refCellQuadraturePoints, quadratureWeights);
    basis.getValues(refCellQuadraturePoints, refCellBasisValues);
    Kokkos::deep_copy(quadraturePoints, Scalar(0.0));  // initialize to 0
    Kokkos::parallel_for(
        Kokkos::RangePolicy<int>(0, numCells),
        LAMBDA_EXPRESSION(int cellOrdinal) {
          for (int ptOrdinal = 0; ptOrdinal < numPoints; ptOrdinal++) {
            int    nodeOrdinal;
            Scalar finalNodeValue = 1.0;
            for (nodeOrdinal = 0; nodeOrdinal < spaceDim; nodeOrdinal++) {
              Scalar nodeValue =
                  refCellQuadraturePoints(ptOrdinal, nodeOrdinal);
              finalNodeValue -= nodeValue;
              for (int d = 0; d < spaceDim; d++) {
                quadraturePoints(cellOrdinal, ptOrdinal, d) +=
                    nodeValue * cellWorkset(cellOrdinal, nodeOrdinal, d);
              }
            }
            nodeOrdinal = spaceDim;
            for (int d = 0; d < spaceDim; d++) {
              quadraturePoints(cellOrdinal, ptOrdinal, d) +=
                  finalNodeValue * cellWorkset(cellOrdinal, nodeOrdinal, d);
            }
          }
        });

    auto x_coords = getArray_Omega_h<double>(
        "forcing function x coords", numCells * numPoints);
    auto y_coords = getArray_Omega_h<double>(
        "forcing function y coords", numCells * numPoints);
    auto z_coords = getArray_Omega_h<double>(
        "forcing function z coords", numCells * numPoints);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<int>(0, numCells),
        LAMBDA_EXPRESSION(int cellOrdinal) {
          int entryOffset = cellOrdinal * numPoints;
          for (int ptOrdinal = 0; ptOrdinal < numPoints; ptOrdinal++) {
            if (spaceDim > 0)
              x_coords[entryOffset + ptOrdinal] =
                  quadraturePoints(cellOrdinal, ptOrdinal, 0);
            if (spaceDim > 1)
              y_coords[entryOffset + ptOrdinal] =
                  quadraturePoints(cellOrdinal, ptOrdinal, 1);
            if (spaceDim > 2)
              z_coords[entryOffset + ptOrdinal] =
                  quadraturePoints(cellOrdinal, ptOrdinal, 2);
          }
        },
        "fill coords");

    Omega_h::ExprReader reader(numCells * numPoints, spaceDim);
    if (spaceDim > 0)
      reader.register_variable("x", Teuchos::any(Omega_h::Reals(x_coords)));
    if (spaceDim > 1)
      reader.register_variable("y", Teuchos::any(Omega_h::Reals(y_coords)));
    if (spaceDim > 2)
      reader.register_variable("z", Teuchos::any(Omega_h::Reals(z_coords)));

    Teuchos::any result;
    reader.read_string(result, _forcingFunctionExpr, "Low Rm forcing function");
    reader.repeat(result);
    auto fxnValues = Teuchos::any_cast<Omega_h::Reals>(result);

    auto rhs = _rhs;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<int>(0, numCells),
        LAMBDA_EXPRESSION(int cellOrdinal) {
          int entryOffset = cellOrdinal * numPoints;

          for (int ptOrdinal = 0; ptOrdinal < numPoints; ptOrdinal++) {
            Scalar fxnValue = fxnValues[entryOffset + ptOrdinal];

            //        cout << "fxnValue(" << cellOrdinal << "," << ptOrdinal << ") = " << fxnValue << endl;

            Omega_h::Matrix<spaceDim, spaceDim> jacobian;

            // compute jacobian/Det/inverse for cell:
            for (int d1 = 0; d1 < spaceDim; d1++) {
              for (int d2 = 0; d2 < spaceDim; d2++) {
                jacobian[d1][d2] = cellWorkset(cellOrdinal, d2, d1) -
                                   cellWorkset(cellOrdinal, spaceDim, d1);
              }
            }
            auto jacobianDet = Omega_h::determinant(jacobian);

            auto weight = quadratureWeights(ptOrdinal) * fabs(jacobianDet);
            for (int fieldOrdinal = 0; fieldOrdinal < numFields;
                 fieldOrdinal++) {
              auto localOrdinal =
                  cells2nodes[cellOrdinal * nodesPerCell + fieldOrdinal];
              // TODO: revise the "0" below if/when we support multiple RHSes
              auto contribution = -weight * fxnValue *
                                  refCellBasisValues(fieldOrdinal, ptOrdinal);
              Kokkos::atomic_add(&rhs(0, localOrdinal), contribution);
            }
          }
        },
        "assemble RHS");
    //    cout << "Completed RHS assembly.\n";
  }

  /*
   Because both MAGMA and ViennaCL apparently assume symmetry even though it's technically
   not required for CG (and they do so in a way that breaks the solve badly), we do make the
   effort here to maintain symmetry while imposing BCs.
   */
  int  numBCs = _bcNodes.size();
  auto rhs = _rhs;
  auto bcNodes = _bcNodes;
  auto bcValues = _bcValues;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<int>(0, numBCs),
      LAMBDA_EXPRESSION(int bcOrdinal) {
        DefaultLocalOrdinal nodeNumber = bcNodes[bcOrdinal];
        Scalar              value = bcValues[bcOrdinal];
        RowMapEntryType     rowStart = rowMap(nodeNumber);
        RowMapEntryType     rowEnd = rowMap(nodeNumber + 1);
        for (RowMapEntryType entryOrdinal = rowStart; entryOrdinal < rowEnd;
             entryOrdinal++) {
          DefaultLocalOrdinal column = columnIndices(entryOrdinal);
          if (column == nodeNumber)  // diagonal
          {
            matrixEntries(entryOrdinal) = 1.0;
          } else {
            // correct the rhs to account for the fact that we'll be zeroing out (col,row) as well
            // to maintain symmetry
            Kokkos::atomic_add(
                &rhs(0, column), -matrixEntries(entryOrdinal) * value);
            matrixEntries(entryOrdinal) = 0.0;
            RowMapEntryType colRowStart = rowMap(column);
            RowMapEntryType colRowEnd = rowMap(column + 1);
            for (RowMapEntryType colRowEntryOrdinal = colRowStart;
                 colRowEntryOrdinal < colRowEnd; colRowEntryOrdinal++) {
              DefaultLocalOrdinal colRowColumn =
                  columnIndices(colRowEntryOrdinal);
              if (colRowColumn == nodeNumber) {
                // this is the (col, row) entry -- clear it, too
                matrixEntries(colRowEntryOrdinal) = 0.0;
              }
            }
          }
        }
      },
      "BC imposition");

  Kokkos::parallel_for(
      Kokkos::RangePolicy<int>(0, numBCs),
      LAMBDA_EXPRESSION(int bcOrdinal) {
        DefaultLocalOrdinal nodeNumber = bcNodes[bcOrdinal];
        Scalar              value = bcValues[bcOrdinal];
        rhs(0, nodeNumber) = value;
      },
      "BC imposition");
}

typedef CrsLinearProblem<
    DefaultLocalOrdinal>
    CrsLinearSolver;

template <int spaceDim>
Teuchos::RCP<CrsLinearSolver> LowRmPotentialSolve<spaceDim>::getDefaultSolver( double tol, 
									       int maxIters) {
  Teuchos::RCP<CrsLinearSolver> solver;
#ifdef HAVE_AMGX
  {
    typedef AmgXSparseLinearProblem<DefaultLocalOrdinal> AmgXLinearProblem;
    std::string configString = AmgXLinearProblem::configurationString(
        AmgXLinearProblem::EAF, tol, maxIters);

    solver = Teuchos::rcp(new AmgXLinearProblem(_matrix, _lhs, _rhs));
  }
#else
  (void)tol;
  (void)maxIters;
#endif

#ifdef HAVE_TPETRA
  if (solver == Teuchos::null) {
    typedef TpetraSparseLinearProblem<Scalar, Ordinal, Layout, Space> TpetraSolver;

    auto tpetraSolver = new TpetraSolver(_matrix, _lhs, _rhs);
    // TODO set tol, maxIters
    solver = Teuchos::rcp(tpetraSolver);
  }
#endif

#ifdef HAVE_VIENNA_CL
  if (solver == Teuchos::null) {
    typedef ViennaSparseLinearProblem<DefaultLocalOrdinal> ViennaSolver;
    auto viennaSolver = new ViennaSolver(_matrix, _lhs, _rhs);
    viennaSolver->setTolerance(tol);
    viennaSolver->setMaxIters(maxIters);
    solver = Teuchos::rcp(viennaSolver);
  }
#endif
  if (solver == Teuchos::null) {
    std::cout << "WARNING: it looks like lgr was built without any "
                 "compatible solvers (AmgX, ViennaCL).  Returning a null "
                 "solver...\n";
  }
  return solver;
}

template <int spaceDim>
void LowRmPotentialSolve<spaceDim>::determineJouleHeating(
    ScalarVector cellInternalEnergy,
    ScalarVector cellJouleEnergy,
    double       V3,
    double       dt,
    bool         warnIfCellInternalEnergyIsEmpty) {
  const int nodesPerCell = spaceDim + 1;

  auto mesh = _meshFields->femesh.omega_h_mesh;
  auto cells2nodes = mesh->ask_elem_verts();

  int numCells = _meshFields->femesh.nelems;

  ScalarVector matrixEntries = _matrix.entries();
  Scalar quadratureWeight = 1.0;  // for a 1-point quadrature rule for simplices
  for (int d = 2; d <= spaceDim; d++) {
    quadratureWeight /= Scalar(d);
  }

  auto coords = mesh->coords();
  auto cellWorkset =
      LAMBDA_EXPRESSION(int cellOrdinal, int nodeOrdinal, int d) {
    DefaultLocalOrdinal vertexNumber =
        cells2nodes[cellOrdinal * nodesPerCell + nodeOrdinal];
    Scalar coord = coords[vertexNumber * spaceDim + d];
    return coord;
  };

  auto conductivity = _conductivity;

  {
    // DEBUGGING
    //    using namespace std;
    //    int elementNumber = 61000;
    //    cout << "element " << elementNumber <<  ", report:\n";
    //    cout << "conductivity = " << _conductivity(elementNumber) << endl;
    //    cout << "user mat ID = " << UserMatID<DefaultFields>()(elementNumber) << endl;
    //    cout << "internal energy = " << cellInternalEnergy(elementNumber) << endl;

    //    cout << "cellOrdinals with mat ID 0: ";
    //    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    //    {
    //      if (UserMatID<DefaultFields>()(cellOrdinal) == 1)
    //      {
    //        cout << cellOrdinal << " ";
    //      }
    //    }
    //    cout << endl;

    //    cout << "cellOrdinals with mat ID 1: ";
    //    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    //    {
    //      if (UserMatID<DefaultFields>()(cellOrdinal) == 1)
    //      {
    //        cout << cellOrdinal << " ";
    //      }
    //    }
    //    cout << endl;
  }

  auto lhs =
      _lhs;  // avoid device issues with "this" by getting a this-less reference to the data
  Scalar integral = 0.0;
  bool   haveStorage = (cellInternalEnergy.size() > 0 && cellJouleEnergy.size() > 0);
  if (!haveStorage && warnIfCellInternalEnergyIsEmpty) {
    std::cout << "WARNING: haveStorage is false.\n";
  }
  Kokkos::parallel_reduce(
      "Joule heating solution integration", numCells,
      LAMBDA_EXPRESSION(int cellOrdinal, Scalar &localIntegral) {
        Omega_h::Matrix<spaceDim, spaceDim> jacobian;
        double                     cellIntegral = 0.0;

        // compute jacobian/Det/inverse for cell:
        for (int d1 = 0; d1 < spaceDim; d1++) {
          for (int d2 = 0; d2 < spaceDim; d2++) {
            jacobian[d1][d2] = cellWorkset(cellOrdinal, d2, d1) -
                               cellWorkset(cellOrdinal, spaceDim, d1);
          }
        }
        auto jacobianDet = Omega_h::determinant(jacobian);
        auto jacobianInverse = Omega_h::invert(jacobian);

        auto cellVolume = fabs(jacobianDet) * quadratureWeight;

        Omega_h::Vector<spaceDim> gradients[nodesPerCell];
        {
          for (int d = 0; d < spaceDim; d++) {
            gradients[spaceDim][d] = 0.0;
          }

          for (int nodeOrdinal = 0; nodeOrdinal < spaceDim;
               nodeOrdinal++)  // "d1" for jacobian
          {
            for (int d = 0; d < spaceDim; d++)  // "d2" for jacobian
            {
              gradients[nodeOrdinal][d] = jacobianInverse[nodeOrdinal][d];
              gradients[spaceDim][d] -= jacobianInverse[nodeOrdinal][d];
            }
          }
        }
        // scalar version:
        auto cellConductivity = conductivity(cellOrdinal);
        // tensor version commented out below
        //    // get conductivity as an Omega_h matrix
        //    const int numSymComponents = symm_ncomps(spaceDim);
        //    using SymTensorVector = Vector<numSymComponents>;
        //    using SymMatrix = Matrix<spaceDim, spaceDim>;
        //
        //    SymTensorVector symTensorVector;
        //    for (int comp=0; comp<numSymComponents; comp++)
        //    {
        //      symTensorVector[comp] = conductivity(cellOrdinal,comp);
        //    }
        //
        //    SymMatrix cellConductivity = vector2symm(symTensorVector);

        //    if (cellConductivity == 0)
        //    {
        //      cout << "WARNING: cell " << cellOrdinal << " has 0 conductivity!\n";
        //    }

        const int column = 0;
        for (int iNode = 0; iNode < nodesPerCell; iNode++) {
          DefaultLocalOrdinal iLocalOrdinal =
              cells2nodes[cellOrdinal * nodesPerCell + iNode];
          Scalar dofValue_i = lhs(column, iLocalOrdinal);

          for (int jNode = 0; jNode < nodesPerCell; jNode++) {
            DefaultLocalOrdinal jLocalOrdinal =
                cells2nodes[cellOrdinal * nodesPerCell + jNode];
            Scalar dofValue_j = lhs(column, jLocalOrdinal);

            Scalar integral_ij =
                (gradients[iNode] * (cellConductivity * gradients[jNode])) *
                cellVolume;
            cellIntegral += integral_ij * dofValue_i * dofValue_j;
          }
        }
        if (haveStorage) {
          Scalar cell_joule_energy = cellIntegral * dt * V3 * V3;
          cellInternalEnergy(cellOrdinal) += cell_joule_energy;
          cellJouleEnergy(cellOrdinal) += cell_joule_energy;
        }
        localIntegral += cellIntegral;
      },
      integral);

  if (haveStorage) {
    _totalJoulesAdded += integral * dt * V3 * V3;
  }

  _K11 =
      integral * _mSeries * _mParallel;  // m for m-fold symmetry (default is 1)
}

template <int spaceDim>
Scalar LowRmPotentialSolve<spaceDim>::getK11() {
  if (_K11 == 0.0) {
    std::cout << "WARNING: getK11() called before getJouleHeating().  It is more "
            "efficient to do it the other way around.\n";
    ScalarVector emptyVector1;
    ScalarVector emptyVector2;

    determineJouleHeating(emptyVector1, emptyVector2, 0.0, 0.0);
  }

  return _K11;
}

template <int spaceDim>
typename LowRmPotentialSolve<spaceDim>::ScalarMultiVector
LowRmPotentialSolve<spaceDim>::getLHS() {
  return _lhs;
}

template <int spaceDim>
CrsMatrixType LowRmPotentialSolve<spaceDim>::getMatrix() {
  return _matrix;
}

template <int spaceDim>
typename LowRmPotentialSolve<spaceDim>::ConductivityType
LowRmPotentialSolve<spaceDim>::getConstantConductivity(
    Scalar constantValue) {
  // sym tensor version commented out:
  //  int numSymComponents = symm_ncomps(spaceDim);
  //  ScalarMultiVector conductivity("conductivity", _meshFields->femesh.nelems, numSymComponents);
  //
  //  auto fillIdentity = LAMBDA_EXPRESSION(DefaultLocalOrdinal elementOrdinal) {
  //    Matrix <spaceDim,spaceDim> matrix = identity_matrix<spaceDim, spaceDim>() * constantValue;
  //    const int numSymComponents = symm_ncomps(spaceDim);
  //
  //    Vector<numSymComponents> symVector = symm2vector(matrix);
  //    for (int i=0; i<numSymComponents; i++)
  //    {
  //      conductivity(elementOrdinal,i) = symVector[i];
  //    }
  //  }; //end lambda fillIdentity
  //  Kokkos::parallel_for( _meshFields->femesh.nelems, fillIdentity );

  LowRmPotentialSolve<spaceDim>::ConductivityType conductivity(
      "conductivity", _meshFields->femesh.nelems);
  Kokkos::deep_copy(conductivity, constantValue);
  return conductivity;
}

template <int spaceDim>
typename LowRmPotentialSolve<spaceDim>::ScalarMultiVector
LowRmPotentialSolve<spaceDim>::getRHS() {
  return _rhs;
}

template <int spaceDim>
Scalar LowRmPotentialSolve<spaceDim>::getTotalJoulesAdded() {
  return _totalJoulesAdded;
}

template <int spaceDim>
bool LowRmPotentialSolve<spaceDim>::haveForcingFunction() {
  return (_forcingFunctionExpr != "");
}

template <int spaceDim>
void LowRmPotentialSolve<spaceDim>::initialize() {
  const int      vertexDim = 0;
  Omega_h::Graph nodeNodeGraph =
      _meshFields->femesh.omega_h_mesh->ask_star(vertexDim);

  auto rowMapOmega_h = nodeNodeGraph.a2ab;
  auto columnIndicesOmega_h = nodeNodeGraph.ab2b;

  auto numRows = rowMapOmega_h.size() - 1;
  auto nnz =
      columnIndicesOmega_h.size() +
      numRows;  // Omega_h does not include the diagonals: add numRows, and then add 1 to each rowMap entry after the first

  //  cout << "numRows: " << numRows << endl;
  //  cout << "nnz: "     << nnz     << endl;
  //
  //  cout << "rowMapOmega_h: [";
  //  for (int i=0; i<numRows; i++)
  //  {
  //    cout << rowMapOmega_h[i] << " ";
  //  }
  //  cout << rowMapOmega_h[numRows] << "]\n";
  //
  //  cout << "columnIndicesOmega_h: [";
  //  for (int i=0; i<columnIndicesOmega_h.size(); i++)
  //  {
  //    cout << columnIndicesOmega_h[i] << " ";
  //  }
  //  cout << "]\n";

  CrsMatrixType::RowMapVector  rowMap("row map", numRows + 1);
  CrsMatrixType::ScalarVector  entries("matrix entries", nnz);
  CrsMatrixType::OrdinalVector columnIndices("column indices", nnz);

  //  auto columnIndicesOmega_h_size = columnIndicesOmega_h.size();

  Kokkos::parallel_for(
      Kokkos::RangePolicy<int>(0, numRows), LAMBDA_EXPRESSION(int rowNumber) {
        auto entryOffset_oh = rowMapOmega_h[rowNumber];
        auto entryOffset = entryOffset_oh + rowNumber;
        rowMap(rowNumber) = entryOffset;
        rowMap(rowNumber + 1) = rowMapOmega_h[rowNumber + 1] + rowNumber + 1;

        auto entryCount = rowMap(rowNumber + 1) - rowMap(rowNumber);
        int  i_oh = 0;  // will track i until we insert the diagonal entry
                        //    if (rowNumber == numRows-1)
                        //    {
        //      cout << "For rowNumber " << rowNumber << ", entryOffset_oh = " << entryOffset_oh;
        //      cout << ", entryOffset = " << entryOffset << ", entryCount = " << entryCount << endl;
        //    }
        for (int i = 0; i < entryCount; i_oh++, i++) {
          //      cout << "columnIndicesOmega_h[" << i_oh + entryOffset_oh << "] = ";
          //      cout << columnIndicesOmega_h[i_oh + entryOffset_oh] << endl;
          bool insertDiagonal = false;
          if ((i_oh == i) &&
              (i_oh + entryOffset_oh >= rowMapOmega_h[rowNumber + 1])) {
            // i_oh == i                    --> have not inserted diagonal
            // i_oh + entryOffset_oh > size --> at the end of the omega_h entries, should insert
            insertDiagonal = true;
          } else if (i_oh == i) {
            // i_oh + entryOffset_oh in bounds
            auto columnIndex = columnIndicesOmega_h[i_oh + entryOffset_oh];
            if (columnIndex > rowNumber) {
              insertDiagonal = true;
            }
          }
          //      cout << "i = " << i << ", i_oh = " << i_oh << endl;
          if (insertDiagonal) {
            // store the diagonal entry
            columnIndices(i + entryOffset) = rowNumber;
            i_oh--;  // i_oh lags i by 1 after we hit the diagonal
          } else {
            columnIndices(i + entryOffset) =
                columnIndicesOmega_h[i_oh + entryOffset_oh];
          }
        }
      });

  //  cout << "rowMap: [";
  //  for (int i=0; i<numRows; i++)
  //  {
  //    cout << rowMap[i] << " ";
  //  }
  //  cout << rowMap[numRows] << "]\n";
  //
  //  cout << "columnIndices: [";
  //  for (int i=0; i<columnIndices.size(); i++)
  //  {
  //    cout << columnIndices[i] << " ";
  //  }
  //  cout << "]\n";

  _matrix = CrsMatrixType(rowMap, columnIndices, entries);

  int numSolves = _numConductors + 1;  // +1 : particular solve
  _lhs = ScalarMultiVector("solution", numSolves, numRows);
  _rhs = ScalarMultiVector("load", numSolves, numRows);
}

template <int spaceDim>
void LowRmPotentialSolve<spaceDim>::initializeCellWorkset(
    CellWorkset &cellWorkset) {
  // TODO: it would be nice to rewrite the CellTools guys that ask for a cellWorkset to use Omega_h's data for this, rather than duplicating here.  As Dan Ibanez points out, this results in a wasteful storage cost -- about 24x in 3D, on average.
  const int nodesPerCell = spaceDim + 1;
  auto      mesh = _meshFields->femesh.omega_h_mesh;
  auto      coords = mesh->coords();
  auto      cells2nodes = mesh->ask_elem_verts();
  auto      numCells = cellWorkset.extent(0);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<int>(0, numCells),
      LAMBDA_EXPRESSION(int cellOrdinal) {
        for (int nodeOrdinal = 0; nodeOrdinal < nodesPerCell; nodeOrdinal++) {
          DefaultLocalOrdinal vertexNumber =
              cells2nodes[cellOrdinal * nodesPerCell + nodeOrdinal];
          for (int d = 0; d < spaceDim; d++) {
            cellWorkset(cellOrdinal, nodeOrdinal, d) =
                coords[vertexNumber * spaceDim + d];
          }
        }
      },
      "initialize workset");
}

template <int spaceDim>
void LowRmPotentialSolve<spaceDim>::resetMesh(
    Teuchos::RCP<DefaultFields> meshFields) {
  _meshFields = meshFields;
  _bcNodes = LocalOrdinalVector("BC nodes", 0);
  _bcValues = ScalarVector("BC values", 0);
}

template <int spaceDim>
void LowRmPotentialSolve<spaceDim>::setBC(
    const std::string &bcExpr,
    const Omega_h::LOs localNodeOrdinals,
    bool               addToExisting) {
  auto numBCs = localNodeOrdinals.size();

  auto x_coords = getArray_Omega_h<double>("BC: x coords", numBCs);
  auto y_coords = getArray_Omega_h<double>("BC: y coords", numBCs);
  auto z_coords = getArray_Omega_h<double>("BC: z coords", numBCs);

  auto mesh = _meshFields->femesh.omega_h_mesh;
  auto coords = mesh->coords();
  auto cells2nodes = mesh->ask_elem_verts();

  Kokkos::parallel_for(
      Kokkos::RangePolicy<int>(0, numBCs),
      LAMBDA_EXPRESSION(int bcOrdinal) {
        DefaultLocalOrdinal vertexNumber = localNodeOrdinals[bcOrdinal];
        int                 entryOffset = vertexNumber * spaceDim;
        if (spaceDim > 0) x_coords[bcOrdinal] = coords[entryOffset + 0];
        if (spaceDim > 1) y_coords[bcOrdinal] = coords[entryOffset + 1];
        if (spaceDim > 2) z_coords[bcOrdinal] = coords[entryOffset + 2];
      },
      "fill BC coords");

  Omega_h::ExprReader reader(numBCs, spaceDim);
  if (spaceDim > 0)
    reader.register_variable("x", Teuchos::any(Omega_h::Reals(x_coords)));
  if (spaceDim > 1)
    reader.register_variable("y", Teuchos::any(Omega_h::Reals(y_coords)));
  if (spaceDim > 2)
    reader.register_variable("z", Teuchos::any(Omega_h::Reals(z_coords)));

  Teuchos::any result;
  reader.read_string(result, bcExpr, "Low Rm bc expression");
  reader.repeat(result);
  auto bcValues = Teuchos::any_cast<Omega_h::Reals>(result);

  setBC(localNodeOrdinals, bcValues, addToExisting);
}

template <int spaceDim>
void LowRmPotentialSolve<spaceDim>::setBC(
    const Omega_h::LOs   localNodeOrdinals,
    const Omega_h::Reals values,
    bool                 addToExisting) {
  TEUCHOS_TEST_FOR_EXCEPTION(
      localNodeOrdinals.size() != values.size(), std::invalid_argument,
      "localOrdinals must be of the same length as values.  "
       << localNodeOrdinals.size() << " != " << values.size() << '\n');
  auto numBCs = values.size();
  int  bcOffset = 0;
  if (!addToExisting) {
    _bcNodes = LocalOrdinalVector("BC nodes", numBCs);
    _bcValues = ScalarVector("BC values", numBCs);
  } else {
    bcOffset = _bcNodes.size();
    Kokkos::resize(_bcNodes, bcOffset + numBCs);
    Kokkos::resize(_bcValues, bcOffset + numBCs);
  }
  // local copies of the View objects (avoids refs to this in lambda)
  auto bcNodes = _bcNodes;
  auto bcValues = _bcValues;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<int>(0, numBCs),
      LAMBDA_EXPRESSION(int nodeOrdinal) {
        bcNodes(nodeOrdinal + bcOffset) = localNodeOrdinals[nodeOrdinal];
        bcValues(nodeOrdinal + bcOffset) = values[nodeOrdinal];
      },
      "set BCs");
}

template <int spaceDim>
void LowRmPotentialSolve<spaceDim>::setForcingFunctionExpr(
    const std::string &forcingFunctionExpr, int degreeForQuadrature) {
  _forcingFunctionExpr = forcingFunctionExpr;
  _quadratureDegreeForForcing = degreeForQuadrature;
}

template <int spaceDim>
void LowRmPotentialSolve<spaceDim>::setConductivity(
    ConductivityType conductivity) {
  _conductivity = conductivity;
}

template <int spaceDim>
void LowRmPotentialSolve<spaceDim>::setPorts(
    const MeshIO &meshIO,
    std::string & inputNodeSetName,
    std::string & outputNodeSetName) {
  auto &nodesets = meshIO.mesh_sets[Omega_h::NODE_SET];

  Omega_h::LOs inputLocalOrdinals, outputLocalOrdinals;
  if (inputNodeSetName != "") {
    auto nsInputIter = nodesets.find(inputNodeSetName);
    LGR_THROW_IF(
        nsInputIter == nodesets.end(),
        "node set " << inputNodeSetName << " doesn't exist!\n");
    inputLocalOrdinals = nsInputIter->second;
  }

  if (outputNodeSetName != "") {
    auto nsOutputIter = nodesets.find(outputNodeSetName);
    LGR_THROW_IF(
        nsOutputIter == nodesets.end(),
        "node set " << outputNodeSetName << " doesn't exist!\n");
    outputLocalOrdinals = nsOutputIter->second;
  }

  Omega_h::LO inputNodeCount = 0;
  if (inputLocalOrdinals.exists()) {
    inputNodeCount = inputLocalOrdinals.size();
  }
  Omega_h::LO outputNodeCount = 0;
  if (outputLocalOrdinals.exists()) {
    outputNodeCount = outputLocalOrdinals.size();
  }

  // for BCs, we want 1s at input nodes, and 0s at output nodes
  Omega_h::Write<Omega_h::LO>   dirichletNodes(inputNodeCount + outputNodeCount);
  Omega_h::Write<Scalar> dirichletValues(inputNodeCount + outputNodeCount);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<int>(0, inputNodeCount),
      LAMBDA_EXPRESSION(int inputNodeOrdinal) {
        dirichletNodes[inputNodeOrdinal] = inputLocalOrdinals[inputNodeOrdinal];
        dirichletValues[inputNodeOrdinal] = 1.0;
      },
      "input node BCs");

  auto mSeries = _mSeries;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<int>(0, outputNodeCount),
      LAMBDA_EXPRESSION(int outputNodeOrdinal) {
        dirichletNodes[outputNodeOrdinal + inputNodeCount] =
            outputLocalOrdinals[outputNodeOrdinal];
        dirichletValues[outputNodeOrdinal + inputNodeCount] =
            1.0 -
            (1.0 / Scalar(mSeries));  // for _mSeries = 1 (default), this is 0.0
      },
      "output node BCs");

  setBC(dirichletNodes, dirichletValues);
}

template <int spaceDim>
void LowRmPotentialSolve<spaceDim>::setUseMFoldSymmetry(
    int mSeries, int mParallel) {
  _mSeries = mSeries;
  _mParallel = mParallel;
}

// explicit instantiation:
template class LowRmPotentialSolve<1>;
template class LowRmPotentialSolve<2>;
template class LowRmPotentialSolve<3>;

}  // end namespace lgr
