/*
 *  StructuralDynamicsTest.cpp
 *  
 *   Created on: May 15, 2018
 **/

#include <iostream>
#include <fstream>

#include "PlatoTestHelpers.hpp"
 
#include "MatrixIO.hpp"
#include "ImplicitFunctors.hpp"

#include "plato/Mechanics.hpp"
#include "plato/WorksetBase.hpp"
#include "plato/VectorFunction.hpp"
#include "plato/StructuralDynamics.hpp"

#include "Teuchos_UnitTestHarness.hpp"

namespace PlatoUnitTests
{

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, ComputeStateWorkset)
{
    // ****** TEST STATE WORKSET TOOLS ****** //
    const Plato::OrdinalType tSpaceDim = 3;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ******************** SET ELASTOSTATICS' EVALUATION TYPES FOR UNIT TEST ********************
    using ResidualT = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::Residual;
    using JacobianU = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::Jacobian;
    using JacobianX = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::GradientX;
    using JacobianZ = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::GradientZ;
    using StrainT = typename Plato::fad_type_t<Plato::Mechanics<tSpaceDim>, ResidualT::StateScalarType, ResidualT::ConfigScalarType>;

    // ALLOCATE ELASTOSTATICS VECTOR FUNCTION
    Plato::DataMap tDataMap;
    VectorFunction<Plato::Mechanics<tSpaceDim>> tElastostatics(*tMesh, tDataMap);

    // ALLOCATE ELASTOSTATICS RESIDUAL
    Omega_h::MeshSets tMeshSets;
    std::shared_ptr<AbstractVectorFunction<ResidualT>> tResidual;
    tResidual = std::make_shared<Plato::ElastostaticResidual<ResidualT, SIMP>>(*tMesh, tMeshSets, tDataMap);
    std::shared_ptr<AbstractVectorFunction<JacobianU>> tJacobianState;
    tJacobianState = std::make_shared<Plato::ElastostaticResidual<JacobianU, SIMP>>(*tMesh, tMeshSets, tDataMap);
    tElastostatics.allocateResidual(tResidual, tJacobianState);

    // SET PROBLEM-RELATED DIMENSIONS
    Plato::OrdinalType tNumCells = tMesh.get()->nelems();
    Plato::OrdinalType tNumVertices = tMesh.get()->nverts();
    Plato::OrdinalType tTotalNumDofs = tNumVertices * tSpaceDim;
    
    // ALLOCATE STATES VECTOR FOR ELASTODYNAMICS EXAMPLE
    Plato::ScalarVector tStateReal("Real States", tTotalNumDofs);
    Plato::ScalarVector tStateImag("Imag States", tTotalNumDofs);
    auto tHostStateReal = Kokkos::create_mirror(tStateReal);
    auto tHostStateImag = Kokkos::create_mirror(tStateImag);
    for(Plato::OrdinalType tIndex = 0; tIndex < tTotalNumDofs; tIndex++)
    {
        tHostStateReal(tIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex);
        tHostStateImag(tIndex) = static_cast<Plato::Scalar>(2e-3) * static_cast<Plato::Scalar>(tIndex);
    }
    Kokkos::deep_copy(tStateReal, tHostStateReal);
    Kokkos::deep_copy(tStateImag, tHostStateImag);
    
    // ALLOCATE STATE WORKSET FOR ELASTODYNAMICS EXAMPLE
    Plato::OrdinalType tNumNodesPerCell = tSpaceDim + static_cast<Plato::OrdinalType>(1);
    Plato::OrdinalType tNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    Plato::ScalarMultiVectorT<ResidualT::StateScalarType> tStateRealWS("Real States Workset", tNumCells, tNumDofsPerCell);
    tElastostatics.worksetState(tStateReal, tStateRealWS);
    Plato::ScalarMultiVectorT<ResidualT::StateScalarType> tStateImagWS("Imag States Workset", tNumCells, tNumDofsPerCell);
    tElastostatics.worksetState(tStateImag, tStateImagWS);

    // ******************** SET ELASTODYNAMICS' EVALUATION TYPES FOR UNIT TEST ********************
    using SD_ResidualT = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::Residual;
    using SD_JacobianU = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::Jacobian;
    using SD_JacobianX = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::GradientX;
    using SD_JacobianZ = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::GradientZ;
    using SD_StrainT = typename
        Plato::fad_type_t<Plato::StructuralDynamics<tSpaceDim>, ResidualT::StateScalarType, ResidualT::ConfigScalarType>;

    // ALLOCATE ELASTODYNAMICS RESIDUAL 
    VectorFunction<Plato::StructuralDynamics<tSpaceDim>> tElastodynamics(*tMesh, tDataMap);
    std::shared_ptr<AbstractVectorFunction<SD_ResidualT>> tResidualSD;
    tResidualSD = std::make_shared<Plato::StructuralDynamicsResidual<SD_ResidualT, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);
    std::shared_ptr<AbstractVectorFunction<SD_JacobianU>> tJacobianStateSD;
    tJacobianStateSD = std::make_shared<Plato::StructuralDynamicsResidual<SD_JacobianU, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);
    tElastodynamics.allocateResidual(tResidualSD, tJacobianStateSD);

    tTotalNumDofs = static_cast<Plato::OrdinalType>(2) * tNumVertices * tSpaceDim;
    tNumDofsPerCell = static_cast<Plato::OrdinalType>(2) * tSpaceDim * tNumNodesPerCell;
    Plato::ScalarVector tComplexStates("ComplexStates", tTotalNumDofs);

    // ALLOCATE STATE VECTOR FOR ELASTODYNAMICS EXAMPLE
    auto tHostComplexStates = Kokkos::create_mirror(tComplexStates);
    const Plato::OrdinalType tNumRealDofs = tNumVertices * tSpaceDim;
    const Plato::OrdinalType tNumDofsPerNode = static_cast<Plato::OrdinalType>(2) * tSpaceDim;
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumRealDofs; tIndex++)
    {
        Plato::OrdinalType tMyRealIndex = (tIndex % tSpaceDim)
            + (static_cast<Plato::OrdinalType>(tIndex/tSpaceDim) * tNumDofsPerNode);
        tHostComplexStates(tMyRealIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex);
        Plato::OrdinalType tMyImagIndex = (tIndex % tSpaceDim) + tSpaceDim
            + (static_cast<Plato::OrdinalType>(tIndex/tSpaceDim) * tNumDofsPerNode);
        tHostComplexStates(tMyImagIndex) = static_cast<Plato::Scalar>(2e-3) * static_cast<Plato::Scalar>(tIndex);
    }
    Kokkos::deep_copy(tComplexStates, tHostComplexStates);

    // ALLOCATE STATE WORKSET FOR ELASTOSTATICS EXAMPLE
    Plato::ScalarMultiVectorT<SD_ResidualT::StateScalarType> tComplexStatesWS("ComplexStatesWS", tNumCells, tNumDofsPerCell);
    tElastodynamics.worksetState(tComplexStates, tComplexStatesWS);

    // TEST WORKSET OUTPUTS
    auto tHostStateRealWS = Kokkos::create_mirror(tStateRealWS);
    Kokkos::deep_copy(tHostStateRealWS, tStateRealWS);
    auto tHostStateImagWS = Kokkos::create_mirror(tStateImagWS);
    Kokkos::deep_copy(tHostStateImagWS, tStateImagWS);
    auto tHostComplexStatesWS = Kokkos::create_mirror(tComplexStatesWS);
    Kokkos::deep_copy(tHostComplexStatesWS, tComplexStatesWS);

    const Plato::Scalar tTolerance = 1e-6;
    const Plato::OrdinalType tNumRealDofsPerCell = tSpaceDim * tNumNodesPerCell;
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumRealDofsPerCell; tDofIndex++)
        {
            Plato::OrdinalType tMyRealIndex = (tDofIndex % tSpaceDim)
                + (static_cast<Plato::OrdinalType>(tDofIndex/tSpaceDim) * tNumDofsPerNode);
            TEST_FLOATING_EQUALITY(tHostComplexStatesWS(tCellIndex, tMyRealIndex), tHostStateRealWS(tCellIndex, tDofIndex), tTolerance);
            Plato::OrdinalType tMyImagIndex = (tDofIndex % tSpaceDim) + tSpaceDim
                + (static_cast<Plato::OrdinalType>(tDofIndex/tSpaceDim) * tNumDofsPerNode);
            TEST_FLOATING_EQUALITY(tHostComplexStatesWS(tCellIndex, tMyImagIndex), tHostStateImagWS(tCellIndex, tDofIndex), tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, CompareLinearStrainsToComplexStrains)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 3;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ******************** SET ELASTOSTATICS' EVALUATION TYPES FOR UNIT TEST ********************
    using ResidualT = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::Residual;
    using JacobianU = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::Jacobian;
    using JacobianX = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::GradientX;
    using JacobianZ = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::GradientZ;
    using StrainT = typename Plato::fad_type_t<Plato::Mechanics<tSpaceDim>, ResidualT::StateScalarType, ResidualT::ConfigScalarType>;
    
    // ALLOCATE ELASTOSTATICS VECTOR FUNCTION
    Plato::DataMap tDataMap;
    VectorFunction<Plato::Mechanics<tSpaceDim>> tElastostatics(*tMesh, tDataMap);

    // ALLOCATE ELASTOSTATICS RESIDUAL
    Omega_h::MeshSets tMeshSets;
    std::shared_ptr<AbstractVectorFunction<ResidualT>> tResidual;
    tResidual = std::make_shared<Plato::ElastostaticResidual<ResidualT, SIMP>>(*tMesh, tMeshSets, tDataMap);
    std::shared_ptr<AbstractVectorFunction<JacobianU>> tJacobianState;
    tJacobianState = std::make_shared<Plato::ElastostaticResidual<JacobianU, SIMP>>(*tMesh, tMeshSets, tDataMap);
    tElastostatics.allocateResidual(tResidual, tJacobianState);

    // SET PROBLEM-RELATED DIMENSIONS
    Plato::OrdinalType tNumCells = tMesh.get()->nelems();
    TEST_EQUALITY(tNumCells, static_cast<Plato::OrdinalType>(6));
    Plato::OrdinalType tNumVertices = tMesh.get()->nverts();
    TEST_EQUALITY(tNumVertices, static_cast<Plato::OrdinalType>(8));
    Plato::OrdinalType tTotalNumDofs = tNumVertices * tSpaceDim;
    TEST_EQUALITY(tTotalNumDofs, static_cast<Plato::OrdinalType>(24));

    // ALLOCATE STATES VECTOR FOR ELASTOSTATICS EXAMPLE
    Plato::ScalarVector tRealStates("Real LinearStates", tTotalNumDofs);
    auto tHostRealStates = Kokkos::create_mirror(tRealStates);
    Plato::ScalarVector tImagStates("Imag LinearStates", tTotalNumDofs);
    auto tHostImagStates = Kokkos::create_mirror(tImagStates);
    for(Plato::OrdinalType tIndex = 0; tIndex < tTotalNumDofs; tIndex++)
    {
        tHostRealStates(tIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex);
        tHostImagStates(tIndex) = static_cast<Plato::Scalar>(2e-3) * static_cast<Plato::Scalar>(tIndex);
    }    
    Kokkos::deep_copy(tRealStates, tHostRealStates);
    Kokkos::deep_copy(tImagStates, tHostImagStates);

    // ALLOCATE STATE WORKSET FOR ELASTOSTATICS EXAMPLE
    Plato::OrdinalType tNumNodesPerCell = tSpaceDim + static_cast<Plato::OrdinalType>(1);
    Plato::OrdinalType tNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    Plato::ScalarMultiVectorT<ResidualT::StateScalarType> tRealStatesWS("Real LinearStates Workset", tNumCells, tNumDofsPerCell);
    tElastostatics.worksetState(tRealStates, tRealStatesWS);
    Plato::ScalarMultiVectorT<ResidualT::StateScalarType> tImagStatesWS("Imag LinearStates Workset", tNumCells, tNumDofsPerCell);
    tElastostatics.worksetState(tImagStates, tImagStatesWS);

    // ALLOCATE COMMON DATA STRUCTURES FOR ELASTOSTATICS AND ELASTODYNAMICS EXAMPLES
    Plato::ComputeGradientWorkset<tSpaceDim> tComputeGradient;
    Plato::ScalarVectorT<ResidualT::ConfigScalarType> tCellVolume("Cell Volume", tNumCells);
    Plato::ScalarArray3DT<ResidualT::ConfigScalarType> tGradient("Gradient", tNumCells, tNumNodesPerCell, tSpaceDim);
    Plato::ScalarArray3DT<ResidualT::ConfigScalarType> tConfigWS("Configuration Workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tElastostatics.worksetConfig(tConfigWS);

    // COMPUTE LINEAR STRAINS 
    Strain<tSpaceDim> tComputeLinearStrain;
    const Plato::OrdinalType tNumVoigtTerms = 6;
    Plato::ScalarMultiVectorT<StrainT> tRealLinearStrain("RealLinearStrain", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<StrainT> tImagLinearStrain("ImagLinearStrain", tNumCells, tNumVoigtTerms);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        // compute strain
        tComputeLinearStrain(aCellOrdinal, tRealLinearStrain, tRealStatesWS, tGradient);
        tComputeLinearStrain(aCellOrdinal, tImagLinearStrain, tImagStatesWS, tGradient);
    }, "UnitTest::LinearStrains");

    // ******************** SET ELASTODYNAMICS' EVALUATION TYPES FOR UNIT TEST ********************
    using SD_ResidualT = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::Residual;
    using SD_JacobianU = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::Jacobian;
    using SD_JacobianX = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::GradientX;
    using SD_JacobianZ = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::GradientZ;
    using SD_StrainT = typename
        Plato::fad_type_t<Plato::StructuralDynamics<tSpaceDim>, ResidualT::StateScalarType, ResidualT::ConfigScalarType>;
    
    // ALLOCATE ELASTODYNAMICS VECTOR FUNCTION
    VectorFunction<Plato::StructuralDynamics<tSpaceDim>> tElastodynamics(*tMesh, tDataMap);
    std::shared_ptr<AbstractVectorFunction<SD_ResidualT>> tResidualSD;
    tResidualSD = std::make_shared<Plato::StructuralDynamicsResidual<SD_ResidualT, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);
    std::shared_ptr<AbstractVectorFunction<SD_JacobianU>> tJacobianStateSD;
    tJacobianStateSD = std::make_shared<Plato::StructuralDynamicsResidual<SD_JacobianU, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);
    tElastodynamics.allocateResidual(tResidualSD, tJacobianStateSD);

    // ALLOCATE STATE VECTOR FOR ELASTODYNAMICS EXAMPLE
    tTotalNumDofs = static_cast<Plato::OrdinalType>(2) * tNumVertices * tSpaceDim;
    TEST_EQUALITY(tTotalNumDofs, static_cast<Plato::OrdinalType>(48));
    tNumDofsPerCell = static_cast<Plato::OrdinalType>(2) * tSpaceDim * tNumNodesPerCell;
    TEST_EQUALITY(tNumDofsPerCell, static_cast<Plato::OrdinalType>(24));
    Plato::ScalarVector tComplexStates("ComplexStates", tTotalNumDofs);

    auto tHostComplexStates = Kokkos::create_mirror(tComplexStates);
    const Plato::OrdinalType tNumRealDofs = tNumVertices * tSpaceDim;
    const Plato::OrdinalType tNumDofsPerNode = static_cast<Plato::OrdinalType>(2) * tSpaceDim;
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumRealDofs; tIndex++)
    {
        Plato::OrdinalType tMyIndex = (tIndex % tSpaceDim)
            + (static_cast<Plato::OrdinalType>(tIndex/tSpaceDim) * tNumDofsPerNode);
        tHostComplexStates(tMyIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex);
        tMyIndex = (tIndex % tSpaceDim) + tSpaceDim + (static_cast<Plato::OrdinalType>(tIndex/tSpaceDim) * tNumDofsPerNode);
        tHostComplexStates(tMyIndex) = static_cast<Plato::Scalar>(2e-3) * static_cast<Plato::Scalar>(tIndex);
    }
    Kokkos::deep_copy(tComplexStates, tHostComplexStates);
    
    // ALLOCATE STATE WORKSET FOR ELASTOSTATICS EXAMPLE
    Plato::ScalarMultiVectorT<SD_ResidualT::StateScalarType> tComplexStatesWS("ComplexStatesWS", tNumCells, tNumDofsPerCell);
    tElastodynamics.worksetState(tComplexStates, tComplexStatesWS);
    
    // COMPUTE COMPLEX STRAINS
    const Plato::OrdinalType tCOMPLEX_SPACE_DIM = 2;
    Plato::ComplexStrain<tSpaceDim, tNumDofsPerNode> tComputeComplexStrain;
    Plato::ScalarArray3DT<StrainT> tComplexStrain("ComplexStrain", tNumCells, tCOMPLEX_SPACE_DIM, tNumVoigtTerms);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tCellVolume(aCellOrdinal) = 0.0;
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        // compute strain
        tComputeComplexStrain(aCellOrdinal, tComplexStatesWS, tGradient, tComplexStrain);
    }, "UnitTest::ComplexStrain");

    // TEST OUTPUTS: LINEAR STRAINS AND COMPLEX STRAINS SHOULD BE EQUAL
    auto tHostRealLinearStrain = Kokkos::create_mirror(tRealLinearStrain);
    Kokkos::deep_copy(tHostRealLinearStrain, tRealLinearStrain);
    auto tHostImagLinearStrain = Kokkos::create_mirror(tImagLinearStrain);
    Kokkos::deep_copy(tHostImagLinearStrain, tImagLinearStrain);
    auto tHostComplexStrain = Kokkos::create_mirror(tComplexStrain);
    Kokkos::deep_copy(tHostComplexStrain, tComplexStrain);
    
    const Plato::Scalar tTolerance = 1e-6;
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tVoigtIndex = 0; tVoigtIndex < tNumVoigtTerms; tVoigtIndex++)
        {
            Plato::OrdinalType tCOMPLEX_SPACE_INDEX = 0;
            TEST_FLOATING_EQUALITY(tHostComplexStrain(tCellIndex, tCOMPLEX_SPACE_INDEX, tVoigtIndex), tHostRealLinearStrain(tCellIndex, tVoigtIndex), tTolerance);
            tCOMPLEX_SPACE_INDEX = 1;
            TEST_FLOATING_EQUALITY(tHostComplexStrain(tCellIndex, tCOMPLEX_SPACE_INDEX, tVoigtIndex), tHostImagLinearStrain(tCellIndex, tVoigtIndex), tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, CompareLinearStressToComplexStress)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 3;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ******************** SET ELASTOSTATICS' EVALUATION TYPES FOR UNIT TEST ********************
    using ResidualT = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::Residual;
    using JacobianU = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::Jacobian;
    using JacobianX = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::GradientX;
    using JacobianZ = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::GradientZ;
    using StrainT = typename Plato::fad_type_t<Plato::Mechanics<tSpaceDim>, ResidualT::StateScalarType, ResidualT::ConfigScalarType>;

    // ALLOCATE ELASTOSTATICS VECTOR FUNCTION
    Plato::DataMap tDataMap;
    VectorFunction<Plato::Mechanics<tSpaceDim>> tElastostatics(*tMesh, tDataMap);

    // ALLOCATE ELASTOSTATICS RESIDUAL
    Omega_h::MeshSets tMeshSets;
    std::shared_ptr<AbstractVectorFunction<ResidualT>> tResidual;
    tResidual = std::make_shared<Plato::ElastostaticResidual<ResidualT, SIMP>>(*tMesh, tMeshSets, tDataMap);
    std::shared_ptr<AbstractVectorFunction<JacobianU>> tJacobianState;
    tJacobianState = std::make_shared<Plato::ElastostaticResidual<JacobianU, SIMP>>(*tMesh, tMeshSets, tDataMap);
    tElastostatics.allocateResidual(tResidual, tJacobianState);
    
    // SET PROBLEM-RELATED DIMENSIONS
    Plato::OrdinalType tNumCells = tMesh.get()->nelems();
    TEST_EQUALITY(tNumCells, static_cast<Plato::OrdinalType>(6));
    Plato::OrdinalType tNumVertices = tMesh.get()->nverts();
    TEST_EQUALITY(tNumVertices, static_cast<Plato::OrdinalType>(8));
    Plato::OrdinalType tTotalNumDofs = tNumVertices * tSpaceDim;
    TEST_EQUALITY(tTotalNumDofs, static_cast<Plato::OrdinalType>(24));

    // ALLOCATE STATES VECTOR FOR ELASTOSTATICS EXAMPLE
    Plato::ScalarVector tRealStates("Real LinearStates", tTotalNumDofs);
    auto tHostRealStates = Kokkos::create_mirror(tRealStates);
    Plato::ScalarVector tImagStates("Imag LinearStates", tTotalNumDofs);
    auto tHostImagStates = Kokkos::create_mirror(tImagStates);
    for(Plato::OrdinalType tIndex = 0; tIndex < tTotalNumDofs; tIndex++)
    {
        tHostRealStates(tIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex);
        tHostImagStates(tIndex) = static_cast<Plato::Scalar>(2e-3) * static_cast<Plato::Scalar>(tIndex);
    }
    Kokkos::deep_copy(tRealStates, tHostRealStates);
    Kokkos::deep_copy(tImagStates, tHostImagStates);

    // ALLOCATE STATE WORKSET FOR ELASTOSTATICS EXAMPLE
    Plato::OrdinalType tNumNodesPerCell = tSpaceDim + static_cast<Plato::OrdinalType>(1);
    Plato::OrdinalType tNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    Plato::ScalarMultiVectorT<ResidualT::StateScalarType> tRealStatesWS("Real LinearStates Workset", tNumCells, tNumDofsPerCell);
    tElastostatics.worksetState(tRealStates, tRealStatesWS);
    Plato::ScalarMultiVectorT<ResidualT::StateScalarType> tImagStatesWS("Imag LinearStates Workset", tNumCells, tNumDofsPerCell);
    tElastostatics.worksetState(tImagStates, tImagStatesWS);

    // ALLOCATE COMMON DATA STRUCTURES FOR ELASTOSTATICS AND ELASTODYNAMICS EXAMPLES
    Plato::ComputeGradientWorkset<tSpaceDim> tComputeGradient;
    Plato::ScalarVectorT<ResidualT::ConfigScalarType> tCellVolume("Cell Volume", tNumCells);
    Plato::ScalarArray3DT<ResidualT::ConfigScalarType> tGradient("Gradient", tNumCells, tNumNodesPerCell, tSpaceDim);
    Plato::ScalarArray3DT<ResidualT::ConfigScalarType> tConfigWS("Configuration Workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tElastostatics.worksetConfig(tConfigWS);

    // COMPUTE LINEAR STRESS
    Plato::Scalar tYoungModulus = 1.0;
    Plato::Scalar tPoissonRatio = 0.3;
    Strain<tSpaceDim> tComputeLinearStrain;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMaterialModel(tYoungModulus, tPoissonRatio);
    auto tStiffnessMatrix = tMaterialModel.getStiffnessMatrix();
    LinearStress<tSpaceDim> tComputeLinearStress(tStiffnessMatrix);

    const Plato::OrdinalType tNumVoigtTerms = 6;
    Plato::ScalarMultiVectorT<StrainT> tRealLinearStrain("RealLinearStrain", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<ResidualT::ResultScalarType> tRealLinearStress("RealLinearStress", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<StrainT> tImagLinearStrain("ImagLinearStrain", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<ResidualT::ResultScalarType> tImagLinearStress("ImagLinearStress", tNumCells, tNumVoigtTerms);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);

        // compute strain
        tComputeLinearStrain(aCellOrdinal, tRealLinearStrain, tRealStatesWS, tGradient);
        tComputeLinearStrain(aCellOrdinal, tImagLinearStrain, tImagStatesWS, tGradient);

        // compute stress
        tComputeLinearStress(aCellOrdinal, tRealLinearStress, tRealLinearStrain);
        tComputeLinearStress(aCellOrdinal, tImagLinearStress, tImagLinearStrain);
    }, "UnitTest::LinearStress");

    // ******************** SET ELASTODYNAMICS' EVALUATION TYPES FOR UNIT TEST ********************
    using SD_ResidualT = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::Residual;
    using SD_JacobianU = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::Jacobian;
    using SD_JacobianX = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::GradientX;
    using SD_JacobianZ = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::GradientZ;
    using SD_StrainT = typename
        Plato::fad_type_t<Plato::StructuralDynamics<tSpaceDim>, ResidualT::StateScalarType, ResidualT::ConfigScalarType>;

    // ALLOCATE ELASTODYNAMICS VECTOR FUNCTION
    VectorFunction<Plato::StructuralDynamics<tSpaceDim>> tElastodynamics(*tMesh, tDataMap);
    std::shared_ptr<AbstractVectorFunction<SD_ResidualT>> tResidualSD;
    tResidualSD = std::make_shared<Plato::StructuralDynamicsResidual<SD_ResidualT, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);
    std::shared_ptr<AbstractVectorFunction<SD_JacobianU>> tJacobianStateSD;
    tJacobianStateSD = std::make_shared<Plato::StructuralDynamicsResidual<SD_JacobianU, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);
    tElastodynamics.allocateResidual(tResidualSD, tJacobianStateSD);

    // ALLOCATE STATE VECTOR FOR ELASTODYNAMICS EXAMPLE
    tTotalNumDofs = static_cast<Plato::OrdinalType>(2) * tNumVertices * tSpaceDim;
    TEST_EQUALITY(tTotalNumDofs, static_cast<Plato::OrdinalType>(48));
    tNumDofsPerCell = static_cast<Plato::OrdinalType>(2) * tSpaceDim * tNumNodesPerCell;
    TEST_EQUALITY(tNumDofsPerCell, static_cast<Plato::OrdinalType>(24));
    Plato::ScalarVector tComplexStates("ComplexStates", tTotalNumDofs);

    auto tHostComplexStates = Kokkos::create_mirror(tComplexStates);
    const Plato::OrdinalType tNumRealDofs = tNumVertices * tSpaceDim;
    const Plato::OrdinalType tNumDofsPerNode = static_cast<Plato::OrdinalType>(2) * tSpaceDim;
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumRealDofs; tIndex++)
    {
        Plato::OrdinalType tMyIndex = (tIndex % tSpaceDim)
            + (static_cast<Plato::OrdinalType>(tIndex/tSpaceDim) * tNumDofsPerNode);
        tHostComplexStates(tMyIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex);
        tMyIndex = (tIndex % tSpaceDim) + tSpaceDim + (static_cast<Plato::OrdinalType>(tIndex/tSpaceDim) * tNumDofsPerNode);
        tHostComplexStates(tMyIndex) = static_cast<Plato::Scalar>(2e-3) * static_cast<Plato::Scalar>(tIndex);
    }
    Kokkos::deep_copy(tComplexStates, tHostComplexStates);

    // ALLOCATE STATE WORKSET FOR ELASTOSTATICS EXAMPLE
    Plato::ScalarMultiVectorT<SD_ResidualT::StateScalarType> tComplexStatesWS("ComplexStatesWS", tNumCells, tNumDofsPerCell);
    tElastodynamics.worksetState(tComplexStates, tComplexStatesWS);

    // COMPUTE COMPLEX STRESSES
    const Plato::OrdinalType tCOMPLEX_SPACE_DIM = 2;
    Plato::ComplexStrain<tSpaceDim, tNumDofsPerNode> tComputeComplexStrain;
    Plato::ComplexLinearStress<tSpaceDim, tNumVoigtTerms> tComputeComplexStress(tStiffnessMatrix);
    Plato::ScalarArray3DT<StrainT>
        tComplexStrain("ComplexStrain", tNumCells, tCOMPLEX_SPACE_DIM, tNumVoigtTerms);
    Plato::ScalarArray3DT<ResidualT::ResultScalarType>
        tComplexStress("ComplexStress", tNumCells, tCOMPLEX_SPACE_DIM, tNumVoigtTerms);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tCellVolume(aCellOrdinal) = 0.0;
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        // compute strain
        tComputeComplexStrain(aCellOrdinal, tComplexStatesWS, tGradient, tComplexStrain);
        // compute stress
        tComputeComplexStress(aCellOrdinal, tComplexStrain, tComplexStress);
    }, "UnitTest::ComplexStress");

    // TEST OUTPUTS: LINEAR STRESSES AND COMPLEX STRESSES SHOULD BE EQUAL
    auto tHostRealLinearStress = Kokkos::create_mirror(tRealLinearStress);
    Kokkos::deep_copy(tHostRealLinearStress, tRealLinearStress);
    auto tHostImagLinearStress = Kokkos::create_mirror(tImagLinearStress);
    Kokkos::deep_copy(tHostImagLinearStress, tImagLinearStress);
    auto tHostComplexStress = Kokkos::create_mirror(tComplexStress);
    Kokkos::deep_copy(tHostComplexStress, tComplexStress);

    const Plato::Scalar tTolerance = 1e-6;
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tVoigtIndex = 0; tVoigtIndex < tNumVoigtTerms; tVoigtIndex++)
        {
            Plato::OrdinalType tCOMPLEX_SPACE_INDEX = 0;
            TEST_FLOATING_EQUALITY(tHostComplexStress(tCellIndex, tCOMPLEX_SPACE_INDEX, tVoigtIndex), tHostRealLinearStress(tCellIndex, tVoigtIndex), tTolerance);
            tCOMPLEX_SPACE_INDEX = 1;
            TEST_FLOATING_EQUALITY(tHostComplexStress(tCellIndex, tCOMPLEX_SPACE_INDEX, tVoigtIndex), tHostImagLinearStress(tCellIndex, tVoigtIndex), tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, CompareLinearElasticForcesToComplexElasticForces)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 3;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ******************** SET ELASTOSTATICS' EVALUATION TYPES FOR UNIT TEST ********************
    using ResidualT = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::Residual;
    using JacobianU = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::Jacobian;
    using JacobianX = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::GradientX;
    using JacobianZ = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::GradientZ;
    using StrainT = typename Plato::fad_type_t<Plato::Mechanics<tSpaceDim>, ResidualT::StateScalarType, ResidualT::ConfigScalarType>;

    // ALLOCATE ELASTOSTATICS VECTOR FUNCTION
    Plato::DataMap tDataMap;
    VectorFunction<Plato::Mechanics<tSpaceDim>> tElastostatics(*tMesh, tDataMap);
    
    // ALLOCATE ELASTOSTATICS RESIDUAL
    Omega_h::MeshSets tMeshSets;
    std::shared_ptr<AbstractVectorFunction<ResidualT>> tResidual;
    tResidual = std::make_shared<Plato::ElastostaticResidual<ResidualT, SIMP>>(*tMesh, tMeshSets, tDataMap);
    std::shared_ptr<AbstractVectorFunction<JacobianU>> tJacobianState;
    tJacobianState = std::make_shared<Plato::ElastostaticResidual<JacobianU, SIMP>>(*tMesh, tMeshSets, tDataMap);
    tElastostatics.allocateResidual(tResidual, tJacobianState);

    // SET PROBLEM-RELATED DIMENSIONS
    Plato::OrdinalType tNumCells = tMesh.get()->nelems();
    TEST_EQUALITY(tNumCells, static_cast<Plato::OrdinalType>(6));
    Plato::OrdinalType tNumVertices = tMesh.get()->nverts();
    TEST_EQUALITY(tNumVertices, static_cast<Plato::OrdinalType>(8));
    Plato::OrdinalType tTotalNumDofs = tNumVertices * tSpaceDim;
    TEST_EQUALITY(tTotalNumDofs, static_cast<Plato::OrdinalType>(24));

    // ALLOCATE STATES VECTOR FOR ELASTOSTATICS EXAMPLE
    Plato::ScalarVector tRealStates("Real LinearStates", tTotalNumDofs);
    auto tHostRealStates = Kokkos::create_mirror(tRealStates);
    Plato::ScalarVector tImagStates("Imag LinearStates", tTotalNumDofs);
    auto tHostImagStates = Kokkos::create_mirror(tImagStates);
    for(Plato::OrdinalType tIndex = 0; tIndex < tTotalNumDofs; tIndex++)
    {
        tHostRealStates(tIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex);
        tHostImagStates(tIndex) = static_cast<Plato::Scalar>(2e-3) * static_cast<Plato::Scalar>(tIndex);
    }
    Kokkos::deep_copy(tRealStates, tHostRealStates);
    Kokkos::deep_copy(tImagStates, tHostImagStates);

    // ALLOCATE STATE WORKSET FOR ELASTOSTATICS EXAMPLE
    Plato::OrdinalType tNumNodesPerCell = tSpaceDim + static_cast<Plato::OrdinalType>(1);
    Plato::OrdinalType tNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    Plato::ScalarMultiVectorT<ResidualT::StateScalarType> tRealStatesWS("Real LinearStates Workset", tNumCells, tNumDofsPerCell);
    tElastostatics.worksetState(tRealStates, tRealStatesWS);
    Plato::ScalarMultiVectorT<ResidualT::StateScalarType> tImagStatesWS("Imag LinearStates Workset", tNumCells, tNumDofsPerCell);
    tElastostatics.worksetState(tImagStates, tImagStatesWS);

    // ALLOCATE COMMON DATA STRUCTURES FOR ELASTOSTATICS AND ELASTODYNAMICS EXAMPLES
    Plato::ComputeGradientWorkset<tSpaceDim> tComputeGradient;
    Plato::LinearTetCubRuleDegreeOne<tSpaceDim> tCubatureRule;
    Plato::ScalarVectorT<ResidualT::ConfigScalarType> tCellVolume("Cell Volume", tNumCells);
    Plato::ScalarArray3DT<ResidualT::ConfigScalarType> tGradient("Gradient", tNumCells, tNumNodesPerCell, tSpaceDim);
    Plato::ScalarArray3DT<ResidualT::ConfigScalarType> tConfigWS("Configuration Workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tElastostatics.worksetConfig(tConfigWS);

    // COMPUTE LINEAR ELASTIC FORCES
    Plato::Scalar tYoungModulus = 1.0;
    Plato::Scalar tPoissonRatio = 0.3;
    Strain<tSpaceDim> tComputeLinearStrain;
    StressDivergence<tSpaceDim> tComputeElasticForces;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMaterialModel(tYoungModulus, tPoissonRatio);
    auto tStiffnessMatrix = tMaterialModel.getStiffnessMatrix();
    LinearStress<tSpaceDim> tComputeLinearStress(tStiffnessMatrix);

    const Plato::OrdinalType tNumVoigtTerms = 6;
    Plato::ScalarMultiVectorT<StrainT> tRealLinearStrain("RealLinearStrain", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<ResidualT::ResultScalarType> tRealLinearStress("RealLinearStress", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<ResidualT::ResultScalarType> tRealElasticForces("RealElasticForces", tNumCells, tNumDofsPerCell);
    Plato::ScalarMultiVectorT<StrainT> tImagLinearStrain("ImagLinearStrain", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<ResidualT::ResultScalarType> tImagLinearStress("ImagLinearStress", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<ResidualT::ResultScalarType> tImagElasticForces("ImagElasticForces", tNumCells, tNumDofsPerCell);

    auto tQuadratureWeight = tCubatureRule.getCubWeight();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tCellVolume(aCellOrdinal) = 0.0;
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;
        // compute strain
        tComputeLinearStrain(aCellOrdinal, tRealLinearStrain, tRealStatesWS, tGradient);
        tComputeLinearStrain(aCellOrdinal, tImagLinearStrain, tImagStatesWS, tGradient);
        // compute stress
        tComputeLinearStress(aCellOrdinal, tRealLinearStress, tRealLinearStrain);
        tComputeLinearStress(aCellOrdinal, tImagLinearStress, tImagLinearStrain);
        // compute elastic forces
        tComputeElasticForces(aCellOrdinal, tRealElasticForces, tRealLinearStress, tGradient, tCellVolume);
        tComputeElasticForces(aCellOrdinal, tImagElasticForces, tImagLinearStress, tGradient, tCellVolume);
    }, "UnitTest::ElasticForces");

    // ******************** SET ELASTODYNAMICS' EVALUATION TYPES FOR UNIT TEST ********************
    using SD_ResidualT = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::Residual;
    using SD_JacobianU = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::Jacobian;
    using SD_JacobianX = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::GradientX;
    using SD_JacobianZ = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::GradientZ;
    using SD_StrainT = typename
        Plato::fad_type_t<Plato::StructuralDynamics<tSpaceDim>, ResidualT::StateScalarType, ResidualT::ConfigScalarType>;

    // ALLOCATE ELASTODYNAMICS VECTOR FUNCTION
    VectorFunction<Plato::StructuralDynamics<tSpaceDim>> tElastodynamics(*tMesh, tDataMap);
    std::shared_ptr<AbstractVectorFunction<SD_ResidualT>> tResidualSD;
    tResidualSD = std::make_shared<Plato::StructuralDynamicsResidual<SD_ResidualT, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);
    std::shared_ptr<AbstractVectorFunction<SD_JacobianU>> tJacobianStateSD;
    tJacobianStateSD = std::make_shared<Plato::StructuralDynamicsResidual<SD_JacobianU, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);
    tElastodynamics.allocateResidual(tResidualSD, tJacobianStateSD);

    // ALLOCATE STATE VECTOR FOR ELASTODYNAMICS EXAMPLE
    tTotalNumDofs = static_cast<Plato::OrdinalType>(2) * tNumVertices * tSpaceDim;
    TEST_EQUALITY(tTotalNumDofs, static_cast<Plato::OrdinalType>(48));
    tNumDofsPerCell = static_cast<Plato::OrdinalType>(2) * tSpaceDim * tNumNodesPerCell;
    TEST_EQUALITY(tNumDofsPerCell, static_cast<Plato::OrdinalType>(24));
    Plato::ScalarVector tComplexStates("ComplexStates", tTotalNumDofs);

    auto tHostComplexStates = Kokkos::create_mirror(tComplexStates);
    const Plato::OrdinalType tNumRealDofs = tNumVertices * tSpaceDim;
    const Plato::OrdinalType tNumDofsPerNode = static_cast<Plato::OrdinalType>(2) * tSpaceDim;
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumRealDofs; tIndex++)
    {
        Plato::OrdinalType tMyIndex = (tIndex % tSpaceDim)
            + (static_cast<Plato::OrdinalType>(tIndex/tSpaceDim) * tNumDofsPerNode);
        tHostComplexStates(tMyIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex);
        tMyIndex = (tIndex % tSpaceDim) + tSpaceDim + (static_cast<Plato::OrdinalType>(tIndex/tSpaceDim) * tNumDofsPerNode);
        tHostComplexStates(tMyIndex) = static_cast<Plato::Scalar>(2e-3) * static_cast<Plato::Scalar>(tIndex);
    }
    Kokkos::deep_copy(tComplexStates, tHostComplexStates);

    // ALLOCATE STATE WORKSET FOR ELASTOSTATICS EXAMPLE
    Plato::ScalarMultiVectorT<SD_ResidualT::StateScalarType> tComplexStatesWS("ComplexStatesWS", tNumCells, tNumDofsPerCell);
    tElastodynamics.worksetState(tComplexStates, tComplexStatesWS);

    // COMPUTE COMPLEX ELASTIC FORCES
    const Plato::OrdinalType tCOMPLEX_SPACE_DIM = 2;
    Plato::ComplexStrain<tSpaceDim, tNumDofsPerNode> tComputeComplexStrain;
    Plato::ComplexLinearStress<tSpaceDim, tNumVoigtTerms> tComputeComplexStress(tStiffnessMatrix);
    Plato::ComplexStressDivergence<tSpaceDim, tNumDofsPerNode> tComputeComplexElasticForces;
    Plato::ScalarArray3DT<StrainT>
        tComplexStrain("ComplexStrain", tNumCells, tCOMPLEX_SPACE_DIM, tNumVoigtTerms);
    Plato::ScalarArray3DT<ResidualT::ResultScalarType>
        tComplexStress("ComplexStress", tNumCells, tCOMPLEX_SPACE_DIM, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<ResidualT::ResultScalarType>
        tComplexElasticForces("ComplexElasticForces", tNumCells, tNumDofsPerCell);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tCellVolume(aCellOrdinal) = 0.0;
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;
        // compute strain
        tComputeComplexStrain(aCellOrdinal, tComplexStatesWS, tGradient, tComplexStrain);
        // compute stress
        tComputeComplexStress(aCellOrdinal, tComplexStrain, tComplexStress);
        // compute elastic forces
        tComputeComplexElasticForces(aCellOrdinal, tCellVolume, tGradient, tComplexStress, tComplexElasticForces);
    }, "UnitTest::ComplexElasticForces");

    // TEST OUTPUTS: LINEAR AND COMPLEX ELASTIC FORCES SHOULD BE EQUAL
    auto tHostRealElasticForces = Kokkos::create_mirror(tRealElasticForces);
    Kokkos::deep_copy(tHostRealElasticForces, tRealElasticForces);
    auto tHostImagElasticForces = Kokkos::create_mirror(tImagElasticForces);
    Kokkos::deep_copy(tHostImagElasticForces, tImagElasticForces);
    auto tHostComplexElasticForces = Kokkos::create_mirror(tComplexElasticForces);
    Kokkos::deep_copy(tHostComplexElasticForces, tComplexElasticForces);

    const Plato::Scalar tTolerance = 1e-6;
    const Plato::OrdinalType tRealNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tRealNumDofsPerCell; tDofIndex++)
        {
            Plato::OrdinalType tMyIndex = (tDofIndex % tSpaceDim)
                    + (static_cast<Plato::OrdinalType>(tDofIndex/tSpaceDim) * tNumDofsPerNode);
            TEST_FLOATING_EQUALITY(tHostComplexElasticForces(tCellIndex, tMyIndex), tHostRealElasticForces(tCellIndex, tDofIndex), tTolerance);
            tMyIndex = (tDofIndex % tSpaceDim) + tSpaceDim
                    + (static_cast<Plato::OrdinalType>(tDofIndex/tSpaceDim) * tNumDofsPerNode);
            TEST_FLOATING_EQUALITY(tHostComplexElasticForces(tCellIndex, tMyIndex), tHostImagElasticForces(tCellIndex, tDofIndex), tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, CompareElastostaticsToElastodynamicsResidual)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 3;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ******************** SET ELASTOSTATICS' EVALUATION TYPES FOR UNIT TEST ********************
    using ResidualT = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::Residual;
    using JacobianU = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::Jacobian;
    using JacobianX = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::GradientX;
    using JacobianZ = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::GradientZ;
    using StrainT = typename Plato::fad_type_t<Plato::Mechanics<tSpaceDim>, ResidualT::StateScalarType, ResidualT::ConfigScalarType>;
   
    // ALLOCATE ELASTOSTATICS VECTOR FUNCTION
    Plato::DataMap tDataMap;
    VectorFunction<Plato::Mechanics<tSpaceDim>> tElastostatics(*tMesh, tDataMap);
    
    // ALLOCATE ELASTOSTATICS RESIDUAL
    Omega_h::MeshSets tMeshSets;
    std::shared_ptr<AbstractVectorFunction<ResidualT>> tResidual;
    tResidual = std::make_shared<Plato::ElastostaticResidual<ResidualT, SIMP>>(*tMesh, tMeshSets, tDataMap);
    std::shared_ptr<AbstractVectorFunction<JacobianU>> tJacobianState;
    tJacobianState = std::make_shared<Plato::ElastostaticResidual<JacobianU, SIMP>>(*tMesh, tMeshSets, tDataMap);
    tElastostatics.allocateResidual(tResidual, tJacobianState);

    // SET PROBLEM-RELATED DIMENSIONS
    Plato::OrdinalType tNumCells = tMesh.get()->nelems();
    TEST_EQUALITY(tNumCells, static_cast<Plato::OrdinalType>(6));
    Plato::OrdinalType tNumVertices = tMesh.get()->nverts();
    TEST_EQUALITY(tNumVertices, static_cast<Plato::OrdinalType>(8));
    Plato::OrdinalType tTotalNumDofs = tNumVertices * tSpaceDim;
    TEST_EQUALITY(tTotalNumDofs, static_cast<Plato::OrdinalType>(24));

    // ALLOCATE STATES VECTOR FOR ELASTOSTATICS EXAMPLE
    Plato::ScalarVector tRealStates("Real LinearStates", tTotalNumDofs);
    auto tHostRealStates = Kokkos::create_mirror(tRealStates);
    Plato::ScalarVector tImagStates("Imag LinearStates", tTotalNumDofs);
    auto tHostImagStates = Kokkos::create_mirror(tImagStates);
    for(Plato::OrdinalType tIndex = 0; tIndex < tTotalNumDofs; tIndex++)
    {
        tHostRealStates(tIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex);
        tHostImagStates(tIndex) = static_cast<Plato::Scalar>(2e-3) * static_cast<Plato::Scalar>(tIndex);
    }
    Kokkos::deep_copy(tRealStates, tHostRealStates);
    Kokkos::deep_copy(tImagStates, tHostImagStates);

    // ALLOCATE CONTROL VECTOR FOR EXAMPLE
    Plato::ScalarVector tControl("Control", tNumVertices);
    auto tHostControl = Kokkos::create_mirror(tControl);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumVertices; tIndex++)
    {
        tHostControl(tIndex) = static_cast<Plato::Scalar>(1.0);
    }
    Kokkos::deep_copy(tControl, tHostControl);

    // COMPUTE ELASTOSTATICS RESIDUAL
    auto tRealElastostaticsResidual = tElastostatics.value(tRealStates, tControl);
    Plato::OrdinalType tSize = tRealElastostaticsResidual.size();
    TEST_EQUALITY(tTotalNumDofs, tSize);
    auto tImagElastostaticsResidual = tElastostatics.value(tImagStates, tControl);
    tSize = tImagElastostaticsResidual.size();
    TEST_EQUALITY(tTotalNumDofs, tSize);

    // ******************** SET ELASTODYNAMICS' EVALUATION TYPES FOR UNIT TEST ********************
    using SD_ResidualT = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::Residual;
    using SD_JacobianU = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::Jacobian;
    using SD_JacobianX = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::GradientX;
    using SD_JacobianZ = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::GradientZ;
    using SD_StrainT = typename
        Plato::fad_type_t<Plato::StructuralDynamics<tSpaceDim>, ResidualT::StateScalarType, ResidualT::ConfigScalarType>;

    // ALLOCATE ELASTODYNAMICS VECTOR FUNCTION
    VectorFunction<Plato::StructuralDynamics<tSpaceDim>> tElastodynamics(*tMesh, tDataMap);
    std::shared_ptr<AbstractVectorFunction<SD_ResidualT>> tResidualSD;
    tResidualSD = std::make_shared<Plato::StructuralDynamicsResidual<SD_ResidualT, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);
    std::shared_ptr<AbstractVectorFunction<SD_JacobianU>> tJacobianStateSD;
    tJacobianStateSD = std::make_shared<Plato::StructuralDynamicsResidual<SD_JacobianU, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);
    tElastodynamics.allocateResidual(tResidualSD, tJacobianStateSD);

    // ALLOCATE STATE VECTOR FOR ELASTODYNAMICS EXAMPLE
    tTotalNumDofs = static_cast<Plato::OrdinalType>(2) * tNumVertices * tSpaceDim;
    TEST_EQUALITY(tTotalNumDofs, static_cast<Plato::OrdinalType>(48));
    Plato::ScalarVector tComplexStates("ComplexStates", tTotalNumDofs);

    auto tHostComplexStates = Kokkos::create_mirror(tComplexStates);
    const Plato::OrdinalType tNumRealDofs = tNumVertices * tSpaceDim;
    TEST_EQUALITY(tNumRealDofs, static_cast<Plato::OrdinalType>(24));
    const Plato::OrdinalType tNumDofsPerNode = static_cast<Plato::OrdinalType>(2) * tSpaceDim;
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumRealDofs; tIndex++)
    {
        Plato::OrdinalType tMyIndex = (tIndex % tSpaceDim)
            + (static_cast<Plato::OrdinalType>(tIndex/tSpaceDim) * tNumDofsPerNode);
        tHostComplexStates(tMyIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex);
        tMyIndex = (tIndex % tSpaceDim) + tSpaceDim + (static_cast<Plato::OrdinalType>(tIndex/tSpaceDim) * tNumDofsPerNode);
        tHostComplexStates(tMyIndex) = static_cast<Plato::Scalar>(2e-3) * static_cast<Plato::Scalar>(tIndex);
    }
    Kokkos::deep_copy(tComplexStates, tHostComplexStates);

    // COMPUTE ELASTODYNAMICS RESIDUAL
    auto tElastodynamicsResidual = tElastodynamics.value(tComplexStates, tControl);
    tSize = tElastodynamicsResidual.size();
    TEST_EQUALITY(tTotalNumDofs, tSize); // tTotalNumDofs = 48

    // TEST OUTPUTS: LINEAR AND COMPLEX ELASTIC FORCES SHOULD BE EQUAL
    auto tHostRealElastostaticsResidual = Kokkos::create_mirror(tRealElastostaticsResidual);
    Kokkos::deep_copy(tHostRealElastostaticsResidual, tRealElastostaticsResidual);
    auto tHostImagElastostaticsResidual = Kokkos::create_mirror(tImagElastostaticsResidual);
    Kokkos::deep_copy(tHostImagElastostaticsResidual, tImagElastostaticsResidual);
    auto tHostElastodynamicsResidual = Kokkos::create_mirror(tElastodynamicsResidual);
    Kokkos::deep_copy(tHostElastodynamicsResidual, tElastodynamicsResidual);

    const Plato::Scalar tTolerance = 1e-6;
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumRealDofs; tDofIndex++)
    {
        Plato::OrdinalType tMyIndex = (tDofIndex % tSpaceDim)
                + (static_cast<Plato::OrdinalType>(tDofIndex/tSpaceDim) * tNumDofsPerNode);
        TEST_FLOATING_EQUALITY(tHostElastodynamicsResidual(tMyIndex), tHostRealElastostaticsResidual(tDofIndex), tTolerance);
        tMyIndex = (tDofIndex % tSpaceDim) + tSpaceDim
                + (static_cast<Plato::OrdinalType>(tDofIndex/tSpaceDim) * tNumDofsPerNode);
        TEST_FLOATING_EQUALITY(tHostElastodynamicsResidual(tMyIndex), tHostImagElastostaticsResidual(tDofIndex), tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, CompareElastostaticsToElastodynamicsGradU)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 3;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ******************** SET ELASTOSTATICS' EVALUATION TYPES FOR UNIT TEST ********************
    using ResidualT = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::Residual;
    using JacobianU = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::Jacobian;
    using JacobianX = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::GradientX;
    using JacobianZ = typename Plato::Evaluation<Plato::Mechanics<tSpaceDim>>::GradientZ;
    using StrainT = typename Plato::fad_type_t<Plato::Mechanics<tSpaceDim>, ResidualT::StateScalarType, ResidualT::ConfigScalarType>;

    // ALLOCATE ELASTOSTATICS VECTOR FUNCTION
    Plato::DataMap tDataMap;
    VectorFunction<Plato::Mechanics<tSpaceDim>> tElastostatics(*tMesh, tDataMap);
    
    // ALLOCATE ELASTOSTATICS RESIDUAL
    Omega_h::MeshSets tMeshSets;
    std::shared_ptr<AbstractVectorFunction<ResidualT>> tResidual;
    tResidual = std::make_shared<Plato::ElastostaticResidual<ResidualT, SIMP>>(*tMesh, tMeshSets, tDataMap);
    std::shared_ptr<AbstractVectorFunction<JacobianU>> tJacobianState;
    tJacobianState = std::make_shared<Plato::ElastostaticResidual<JacobianU, SIMP>>(*tMesh, tMeshSets, tDataMap);
    tElastostatics.allocateResidual(tResidual, tJacobianState);

    // SET PROBLEM-RELATED DIMENSIONS
    Plato::OrdinalType tNumCells = tMesh.get()->nelems();
    TEST_EQUALITY(tNumCells, static_cast<Plato::OrdinalType>(6));
    Plato::OrdinalType tNumVertices = tMesh.get()->nverts();
    TEST_EQUALITY(tNumVertices, static_cast<Plato::OrdinalType>(8));
    Plato::OrdinalType tTotalNumDofs = tNumVertices * tSpaceDim;
    TEST_EQUALITY(tTotalNumDofs, static_cast<Plato::OrdinalType>(24));

    // ALLOCATE STATES VECTOR FOR ELASTOSTATICS EXAMPLE
    Plato::ScalarVector tLinearStates("LinearStates", tTotalNumDofs);

    // ALLOCATE CONTROL VECTOR FOR EXAMPLE
    Plato::ScalarVector tControl("Control", tNumVertices);
    auto tHostControl = Kokkos::create_mirror(tControl);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumVertices; tIndex++)
    {
        tHostControl(tIndex) = static_cast<Plato::Scalar>(1.0);
    }
    Kokkos::deep_copy(tControl, tHostControl);

    // COMPUTE ELASTOSTATICS JACOBIAN U
    auto tElastostaticsJacobianU = tElastostatics.gradient_u(tLinearStates, tControl);

    // ******************** SET ELASTODYNAMICS' EVALUATION TYPES FOR UNIT TEST ********************
    using SD_ResidualT = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::Residual;
    using SD_JacobianU = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::Jacobian;
    using SD_JacobianX = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::GradientX;
    using SD_JacobianZ = typename Plato::Evaluation<Plato::StructuralDynamics<tSpaceDim>>::GradientZ;
    using SD_StrainT = typename
        Plato::fad_type_t<Plato::StructuralDynamics<tSpaceDim>, ResidualT::StateScalarType, ResidualT::ConfigScalarType>;

    // ALLOCATE ELASTODYNAMICS VECTOR FUNCTION
    VectorFunction<Plato::StructuralDynamics<tSpaceDim>> tElastodynamics(*tMesh, tDataMap);
    std::shared_ptr<AbstractVectorFunction<SD_ResidualT>> tResidualSD;
    tResidualSD = std::make_shared<Plato::StructuralDynamicsResidual<SD_ResidualT, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);
    std::shared_ptr<AbstractVectorFunction<SD_JacobianU>> tJacobianStateSD;
    tJacobianStateSD = std::make_shared<Plato::StructuralDynamicsResidual<SD_JacobianU, SIMP, Plato::HyperbolicTangentProjection>>(*tMesh, tMeshSets, tDataMap);
    tElastodynamics.allocateResidual(tResidualSD, tJacobianStateSD);

    // ALLOCATE STATE VECTOR FOR ELASTODYNAMICS EXAMPLE
    tTotalNumDofs = static_cast<Plato::OrdinalType>(2) * tNumVertices * tSpaceDim;
    TEST_EQUALITY(tTotalNumDofs, static_cast<Plato::OrdinalType>(48));
    Plato::ScalarVector tComplexStates("ComplexStates", tTotalNumDofs);

    // COMPUTE ELASTODYNAMICS JACOBIAN U
    auto tNumDofsPerNode = static_cast<Plato::OrdinalType>(2) * tSpaceDim;
    auto tElastodynamicsJacobianU = tElastodynamics.gradient_u(tComplexStates, tControl);
 
    // TEST OUTPUTS: LINEAR AND COMPLEX ELASTIC FORCES SHOULD BE EQUAL
    auto tElastostaticsJacEntries = tElastostaticsJacobianU->entries();
    auto tHostElastostaticsJacEntries = Kokkos::create_mirror(tElastostaticsJacEntries);
    Kokkos::deep_copy(tHostElastostaticsJacEntries, tElastostaticsJacEntries);
    
    auto tElastodynamicsJacEntries = tElastodynamicsJacobianU->entries();
    auto tHostElastodynamicsJacEntries = Kokkos::create_mirror(tElastodynamicsJacEntries);
    Kokkos::deep_copy(tHostElastodynamicsJacEntries, tElastodynamicsJacEntries);
    
    const Plato::Scalar tTolerance = 1e-6;
    const Plato::OrdinalType tComplexSpaceStride = tNumDofsPerNode * tSpaceDim; // tNumDofsPerNode = 6
    for(Plato::OrdinalType tVertexIndex = 0; tVertexIndex < tNumVertices; tVertexIndex++)
    {
        auto tElastoStaticsStride = tVertexIndex * (tSpaceDim * tSpaceDim);
        auto tElastoDynamicsStride = tVertexIndex * (tNumDofsPerNode * tNumDofsPerNode); // tNumDofsPerNode = 6
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < tSpaceDim; tDimIndex++)
        {
            auto tMyIndex = tElastoStaticsStride + (tSpaceDim * tDimIndex) + tDimIndex; // tNumDofsPerNode = tSpaceDim
            auto tRealIndex = tElastoDynamicsStride + (tNumDofsPerNode * tDimIndex) + tDimIndex; 
            TEST_FLOATING_EQUALITY(tHostElastodynamicsJacEntries(tRealIndex), tHostElastostaticsJacEntries(tMyIndex), tTolerance);
            auto tImagIndex = tElastoDynamicsStride + tComplexSpaceStride + (tNumDofsPerNode * tDimIndex) + tSpaceDim + tDimIndex; 
            TEST_FLOATING_EQUALITY(tHostElastodynamicsJacEntries(tImagIndex), tHostElastostaticsJacEntries(tMyIndex), tTolerance);
        }
    }
}

} // namespace PlatoUnitTests
