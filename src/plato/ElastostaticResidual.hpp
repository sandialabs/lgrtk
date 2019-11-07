#ifndef ELASTOSTATIC_RESIDUAL_HPP
#define ELASTOSTATIC_RESIDUAL_HPP

#include <memory>

#include "plato/PlatoTypes.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/SimplexMechanics.hpp"
#include "plato/Strain.hpp"
#include "plato/LinearStress.hpp"
#include "plato/StressDivergence.hpp"
#include "plato/AbstractVectorFunction.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/CellForcing.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"
#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"
#include "plato/ToMap.hpp"

#include "plato/LinearElasticMaterial.hpp"
#include "plato/NaturalBCs.hpp"
#include "plato/BodyLoads.hpp"

#include "plato/ExpInstMacros.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Elastostatic vector function interface
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function used for density-based methods
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ElastostaticResidual :
        public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
        public Plato::AbstractVectorFunction<EvaluationType>
{
private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;

    using Plato::SimplexMechanics<mSpaceDim>::mNumVoigtTerms;
    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell;
    using Plato::SimplexMechanics<mSpaceDim>::mNumDofsPerNode;
    using Plato::SimplexMechanics<mSpaceDim>::mNumDofsPerCell;

    using Plato::AbstractVectorFunction<EvaluationType>::mMesh;
    using Plato::AbstractVectorFunction<EvaluationType>::mDataMap;
    using Plato::AbstractVectorFunction<EvaluationType>::mMeshSets;

    using StateScalarType = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;

    Omega_h::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffness;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mSpaceDim, mNumVoigtTerms, IndicatorFunctionType> mApplyWeighting;

    std::shared_ptr<Plato::BodyLoads<EvaluationType>> mBodyLoads;
    std::shared_ptr<Plato::NaturalBCs<mSpaceDim,mNumDofsPerNode>> mBoundaryLoads;
    std::shared_ptr<Plato::CellForcing<mNumVoigtTerms>> mCellForcing;
    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule;

    Teuchos::RCP<Plato::LinearElasticMaterial<mSpaceDim>> mMaterialModel;

    std::vector<std::string> mPlottable;

public:
    /******************************************************************************//**
     * @brief Constructor
     * @param [in] aMesh volume mesh database
     * @param [in] aMeshSets surface mesh database
     * @param [in] aDataMap PLATO Analyze database
     * @param [in] aProblemParams input parameters for overall problem
     * @param [in] aPenaltyParams input parameters for penalty function
    **********************************************************************************/
    ElastostaticResidual(Omega_h::Mesh& aMesh,
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap,
                         Teuchos::ParameterList& aProblemParams,
                         Teuchos::ParameterList& aPenaltyParams) :
            Plato::AbstractVectorFunction<EvaluationType>(aMesh, aMeshSets, aDataMap),
            mIndicatorFunction(aPenaltyParams),
            mApplyWeighting(mIndicatorFunction),
            mBodyLoads(nullptr),
            mBoundaryLoads(nullptr),
            mCellForcing(nullptr),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    {
        // create material model and get stiffness
        //
        Plato::ElasticModelFactory<mSpaceDim> tMaterialModelFactory(aProblemParams);
        mMaterialModel = tMaterialModelFactory.create();

        // parse body loads
        // 
        if(aProblemParams.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType>>(aProblemParams.sublist("Body Loads"));
        }
  
        // parse boundary Conditions
        // 
        if(aProblemParams.isSublist("Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<mSpaceDim,mNumDofsPerNode>>(aProblemParams.sublist("Natural Boundary Conditions"));
        }
  
        // parse cell problem forcing
        //
        if(aProblemParams.isSublist("Cell Problem Forcing"))
        {
            Plato::OrdinalType tColumnIndex = aProblemParams.sublist("Cell Problem Forcing").get<Plato::OrdinalType>("Column Index");
            mCellForcing = std::make_shared<Plato::CellForcing<mNumVoigtTerms>>(mCellStiffness, tColumnIndex);
        }

        auto tResidualParams = aProblemParams.sublist("Elliptic");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
          mPlottable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();

    }

    /******************************************************************************//**
     * @brief Evaluate vector function
     * @param [in] aState 2D array with state variables (C,DOF)
     * @param [in] aControl 2D array with control variables (C,N)
     * @param [in] aConfig 3D array with control variables (C,N,D)
     * @param [in] aResult 1D array with control variables (C,DOF)
     * @param [in] aTimeStep current time step
     * Nomenclature: C = number of cells, DOF = number of degrees of freedom per cell
     * N = number of nodes per cell, D = spatial dimensions
    **********************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> & aState,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarMultiVectorT<ResultScalarType> & aResult,
                  Plato::Scalar aTimeStep = 0.0) const
    {
      auto tNumCells = mMesh.nelems();

      using StrainScalarType =
          typename Plato::fad_type_t<Plato::SimplexMechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
      Plato::Strain<mSpaceDim>                 tComputeVoigtStrain;
      Plato::LinearStress<mSpaceDim>           tComputeVoigtStress(mMaterialModel);
      Plato::StressDivergence<mSpaceDim>       tComputeStressDivergence;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<StrainScalarType>
        tStrain("strain",tNumCells,mNumVoigtTerms);
    
      Plato::ScalarArray3DT<ConfigScalarType>
        tGradient("gradient",tNumCells,mNumNodesPerCell,mSpaceDim);
    
      Plato::ScalarMultiVectorT<ResultScalarType>
        tStress("stress",tNumCells,mNumVoigtTerms);
    
      auto tQuadratureWeight = mCubatureRule->getCubWeight();
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;

        // compute strain
        tComputeVoigtStrain(aCellOrdinal, tStrain, aState, tGradient);
    
        // compute stress
        tComputeVoigtStress(aCellOrdinal, tStress, tStrain);
      }, "Cauchy stress");

      if( mCellForcing != nullptr )
      {
          mCellForcing->add( tStress );
      }

      auto& tApplyWeighting = mApplyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        // apply weighting
        tApplyWeighting(aCellOrdinal, tStress, aControl);
    
        // compute stress divergence
        tComputeStressDivergence(aCellOrdinal, aResult, tStress, tGradient, tCellVolume);
      }, "Apply weighting and compute divergence");

      if( mBodyLoads != nullptr )
      {
          mBodyLoads->get( mMesh, aState, aControl, aResult, -1.0 );
      }

      if( mBoundaryLoads != nullptr )
      {
          mBoundaryLoads->get( &mMesh, mMeshSets, aState, aControl, aConfig, aResult, -1.0 );
      }

      if( std::count(mPlottable.begin(),mPlottable.end(),"strain") ) toMap(mDataMap, tStrain, "strain");
      if( std::count(mPlottable.begin(),mPlottable.end(),"stress") ) toMap(mDataMap, tStress, "stress");

    }
};
// class ElastostaticResidual

} // namespace Plato

#ifdef PLATO_1D
PLATO_EXPL_DEC(Plato::ElastostaticResidual, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEC(Plato::ElastostaticResidual, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEC(Plato::ElastostaticResidual, Plato::SimplexMechanics, 3)
#endif

#endif
