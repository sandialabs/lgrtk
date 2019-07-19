#define block_size_3

//#include "Omega_h_build.hpp"
#include <Omega_h_assoc.hpp>
#include <Omega_h_expr.hpp>
#include <Omega_h_matrix.hpp>

#include "ElastostaticSolve.hpp"
#include "ApplyConstraints.hpp"

#include "CrsMatrix.hpp"

#ifdef HAVE_VIENNA_CL
#include "ViennaSparseLinearProblem.hpp"
#endif

#ifdef HAVE_AMGX
#include "AmgXSparseLinearProblem.hpp"
#include <fstream>
#include <sstream>
#include <string>
#endif

using namespace Plato::LinearElastostatics;

/******************************************************************************/
template<int SpaceDim>
void ElastostaticSolve<SpaceDim>::setBC(const Omega_h::LOs localDofOrdinals,
                                        const Omega_h::Reals values,
                                        bool addToExisting)
/******************************************************************************/
{
    if(localDofOrdinals.size() != values.size())
    {
        std::cout << "localOrdinals must be of the same length as values.  " << localDofOrdinals.size() << " != "
                << values.size() << std::endl;
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localOrdinals must be of the same length as values");
    }
    auto numBCs = values.size();
    int bcOffset = 0;
    if(!addToExisting)
    {
        mBcDofs = Plato::LocalOrdinalVector("BC dofs", numBCs);
        mBcValues = Plato::ScalarVector("BC values", numBCs);
    }
    else
    {
        bcOffset = mBcDofs.size();
        Kokkos::resize(mBcDofs, bcOffset + numBCs);
        Kokkos::resize(mBcValues, bcOffset + numBCs);
    }
    // local copies of the View objects (avoids refs to this in lambda)
    auto bcDofs = mBcDofs;
    auto bcValues = mBcValues;
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0, numBCs), LAMBDA_EXPRESSION(int dofOrdinal)
    {
        bcDofs(dofOrdinal+bcOffset) = localDofOrdinals[dofOrdinal];
        bcValues(dofOrdinal+bcOffset) = values[dofOrdinal];
    },
                         "set BCs");
}

/******************************************************************************/
template<int SpaceDim>
void ElastostaticSolve<SpaceDim>::initialize()
/******************************************************************************/
{

    auto mesh = mMeshFields->femesh.omega_h_mesh;

    Plato::RowMapEntryType numRows;
    if(mUseBlockMatrix)
    {
        mMatrix = Plato::CreateBlockMatrix<CrsMatrixType, SpaceDim>(mesh);
        numRows = SpaceDim * (mMatrix->rowMap().size() - 1);
    }
    else
    {
        mMatrix = Plato::CreateMatrix<CrsMatrixType, SpaceDim>(mesh);
        numRows = mMatrix->rowMap().size() - 1;
    }

    mLhs = Plato::ScalarVector("solution", numRows);
    mRhs = Plato::ScalarVector("load", numRows);

}

/******************************************************************************/
template<int SpaceDim>
ElastostaticSolve<SpaceDim>::ElastostaticSolve(Teuchos::ParameterList const& paramList,
                                               Teuchos::RCP<DefaultFields> meshFields,
                                               Plato::comm::Machine machine) :
        mMachine(machine),
        mMeshFields(meshFields),
        mUseBlockMatrix(paramList.get<bool>("Use Block Matrix"))
/******************************************************************************/
{
    mBcDofs = Plato::LocalOrdinalVector("BC dofs", 0);
    mBcValues = Plato::ScalarVector("BC values", 0);

    Plato::ElasticModelFactory < SpaceDim > mmfactory(paramList);
    mMaterialModel = mmfactory.create();
}

/******************************************************************************/
template<int SpaceDim>
void ElastostaticSolve<SpaceDim>::assemble()
/******************************************************************************/
{
    if(mUseBlockMatrix)
    {
        computeGlobalStiffness<BlockMatrixEntryOrdinal>();
        Plato::applyBlockConstraints<SpaceDim>(mMatrix, mRhs, mBcDofs, mBcValues);
    }
    else
    {
        computeGlobalStiffness<MatrixEntryOrdinal>();
        Plato::applyConstraints<SpaceDim>(mMatrix, mRhs, mBcDofs, mBcValues);
    }

}
/******************************************************************************/
template<int SpaceDim>
template<template<int Dimension = SpaceDim, int DofsPerNode_I = SpaceDim, int DofsPerNode_J = SpaceDim> class LookupType>
void ElastostaticSolve<SpaceDim>::computeGlobalStiffness()
/******************************************************************************/
{
    constexpr int nodesPerCell = SpaceDim + 1;
    constexpr int numDofsPerNode = SpaceDim;
    constexpr int dofsPerCell = nodesPerCell * numDofsPerNode;
    constexpr auto numVoigtTerms = (SpaceDim == 3) ? 6: ((SpaceDim == 2) ? 3: 1);

    auto mesh = mMeshFields->femesh.omega_h_mesh;

    LookupType<SpaceDim, SpaceDim> entryOrdinalLookup(mMatrix, mesh);

    Scalar quadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
    for(int d = 2; d <= SpaceDim; d++)
    {
        quadratureWeight /= Scalar(d);
    }

    // create functor for accessing node coordinates
    //
    Plato::NodeCoordinate<SpaceDim> nodeCoordinate(mesh);

    // create functor for create gradient matrix
    //
    Plato::ComputeGradient<SpaceDim> computeGradient(nodeCoordinate);

    // create functor for creating symmetric gradient matrix
    //
    Plato::ComputeGradientMatrix<SpaceDim> computeGradientMatrix;

    // create stiffness matrix
    //
    auto cellStiffness = mMaterialModel->getStiffnessMatrix();

    // create Assembly functor
    //
    Plato::Assemble<SpaceDim, LookupType<SpaceDim, SpaceDim>> asmbl(cellStiffness, mMatrix, entryOrdinalLookup);

    int numCells = mMeshFields->femesh.nelems;
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0, numCells), LAMBDA_EXPRESSION(int cellOrdinal)
    {

        Scalar cellVolume;
        Omega_h::Vector<SpaceDim> gradients[nodesPerCell];
        computeGradient(cellOrdinal, gradients, cellVolume);
        cellVolume *= quadratureWeight;

        Omega_h::Vector<numVoigtTerms> gradientMatrix[dofsPerCell];
        computeGradientMatrix(gradients, gradientMatrix);

        asmbl(cellOrdinal, gradientMatrix, cellVolume);

    },
                         "grad-grad integration");
}

typedef Plato::CrsLinearProblem<Plato::OrdinalType> CrsLinearSolver;

template<int spaceDim>
Teuchos::RCP<CrsLinearSolver>
#if defined HAVE_AMGX
ElastostaticSolve<spaceDim>::getDefaultSolver(double tol, int maxIters)
#elif defined HAVE_TPETRA
ElastostaticSolve<spaceDim>::getDefaultSolver(double /*tol*/, int /*maxIters*/)
#elif defined HAVE_VIENNA_CL
ElastostaticSolve<spaceDim>::getDefaultSolver(double tol, int maxIters)
#else
ElastostaticSolve<spaceDim>::getDefaultSolver(double /*tol*/, int /*maxIters*/)
#endif
{
    Teuchos::RCP<CrsLinearSolver> solver;

#ifdef HAVE_AMGX
    {

        std::string configString;

        std::ifstream infile;
        infile.open("amgx.json", std::ifstream::in);
        if(infile)
        {
            std::string line;
            std::stringstream config;
            while (std::getline(infile, line))
            {
                std::istringstream iss(line);
                config << iss.str();
            }
            configString = config.str();
        }
        else
        {
            typedef Plato::AmgXSparseLinearProblem<Plato::OrdinalType> AmgXLinearProblem;
            configString = AmgXLinearProblem::configurationString(AmgXLinearProblem::PCG_NOPREC,tol,maxIters);
        }

        if( mUseBlockMatrix )
        {
            typedef Plato::AmgXSparseLinearProblem<Plato::OrdinalType, spaceDim> AmgXLinearProblem;
            solver = Teuchos::rcp(new AmgXLinearProblem(*mMatrix,mLhs,mRhs, configString));
        }
        else
        {
            typedef Plato::AmgXSparseLinearProblem<Plato::OrdinalType> AmgXLinearProblem;
            solver = Teuchos::rcp(new AmgXLinearProblem(*mMatrix,mLhs,mRhs, configString));
        }
    }
#endif
    
#ifdef HAVE_TPETRA
    if (solver == Teuchos::null)
    {
        typedef TpetraSparseLinearProblem<Scalar, Ordinal, DefaultLayout, Space> TpetraSolver;

        auto tpetraSolver = new TpetraSolver(*mMatrix,mLhs,mRhs);
        // TODO set tol, maxIters
        solver = Teuchos::rcp(tpetraSolver);
    }
#endif
    
#ifdef HAVE_VIENNA_CL
    if (solver == Teuchos::null)
    {
        typedef ViennaSparseLinearProblem<Scalar, OrdinalType, DefaultLayout, DefaultSpace> ViennaSolver;
        auto viennaSolver = new ViennaSolver(*mMatrix,mLhs,mRhs);
        viennaSolver->setTolerance(tol);
        viennaSolver->setMaxIters(maxIters);
        solver = Teuchos::rcp(viennaSolver);
    }
#endif

    if(solver == Teuchos::null)
    {
        std::cout
                << "WARNING: it looks like Plato was built without any compatible solvers (AmgX, ViennaCL).  Returning a null solver...\n";
    }
    return solver;

}

// explicit instantiation
template class Plato::LinearElastostatics::ElastostaticSolve<1>;
template class Plato::LinearElastostatics::ElastostaticSolve<2>;
template class Plato::LinearElastostatics::ElastostaticSolve<3>;

