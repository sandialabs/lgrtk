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

    using Plato::SimplexMechanics<mSpaceDim>::m_numVoigtTerms;
    using Simplex<mSpaceDim>::m_numNodesPerCell;
    using Plato::SimplexMechanics<mSpaceDim>::m_numDofsPerNode;
    using Plato::SimplexMechanics<mSpaceDim>::m_numDofsPerCell;

    using Plato::AbstractVectorFunction<EvaluationType>::mMesh;
    using Plato::AbstractVectorFunction<EvaluationType>::m_dataMap;
    using Plato::AbstractVectorFunction<EvaluationType>::mMeshSets;

    using StateScalarType = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;

    Omega_h::Matrix<m_numVoigtTerms, m_numVoigtTerms> m_cellStiffness;

    IndicatorFunctionType m_indicatorFunction;
    Plato::ApplyWeighting<mSpaceDim, m_numVoigtTerms, IndicatorFunctionType> m_applyWeighting;

    std::shared_ptr<Plato::BodyLoads<mSpaceDim,m_numDofsPerNode>> m_bodyLoads;
    std::shared_ptr<Plato::NaturalBCs<mSpaceDim,m_numDofsPerNode>> m_boundaryLoads;
    std::shared_ptr<Plato::CellForcing<m_numVoigtTerms>> m_cellForcing;
    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule;

    std::vector<std::string> m_plottable;

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
            m_indicatorFunction(aPenaltyParams),
            m_applyWeighting(m_indicatorFunction),
            m_bodyLoads(nullptr),
            m_boundaryLoads(nullptr),
            m_cellForcing(nullptr),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    {
        // create material model and get stiffness
        //
        Plato::ElasticModelFactory<mSpaceDim> tMaterialModelFactory(aProblemParams);
        auto tMaterialModel = tMaterialModelFactory.create();
        m_cellStiffness = tMaterialModel->getStiffnessMatrix();
  

        // parse body loads
        // 
        if(aProblemParams.isSublist("Body Loads"))
        {
            m_bodyLoads = std::make_shared<Plato::BodyLoads<mSpaceDim,m_numDofsPerNode>>(aProblemParams.sublist("Body Loads"));
        }
  
        // parse boundary Conditions
        // 
        if(aProblemParams.isSublist("Natural Boundary Conditions"))
        {
            m_boundaryLoads = std::make_shared<Plato::NaturalBCs<mSpaceDim,m_numDofsPerNode>>(aProblemParams.sublist("Natural Boundary Conditions"));
        }
  
        // parse cell problem forcing
        //
        if(aProblemParams.isSublist("Cell Problem Forcing"))
        {
            Plato::OrdinalType tColumnIndex = aProblemParams.sublist("Cell Problem Forcing").get<Plato::OrdinalType>("Column Index");
            m_cellForcing = std::make_shared<Plato::CellForcing<m_numVoigtTerms>>(m_cellStiffness, tColumnIndex);
        }

        auto tResidualParams = aProblemParams.sublist("Elastostatics");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
          m_plottable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();

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
      Strain<mSpaceDim>                        tComputeVoigtStrain;
      LinearStress<mSpaceDim>                  tComputeVoigtStress(m_cellStiffness);
      StressDivergence<mSpaceDim>              tComputeStressDivergence;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<StrainScalarType>
        tStrain("strain",tNumCells,m_numVoigtTerms);
    
      Plato::ScalarArray3DT<ConfigScalarType>
        tGradient("gradient",tNumCells,m_numNodesPerCell,mSpaceDim);
    
      Plato::ScalarMultiVectorT<ResultScalarType>
        tStress("stress",tNumCells,m_numVoigtTerms);
    
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

      if( m_cellForcing != nullptr )
      {
          m_cellForcing->add( tStress );
      }

      auto& tApplyWeighting = m_applyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        // apply weighting
        tApplyWeighting(aCellOrdinal, tStress, aControl);
    
        // compute stress divergence
        tComputeStressDivergence(aCellOrdinal, aResult, tStress, tGradient, tCellVolume);
      }, "Apply weighting and compute divergence");

      if( m_bodyLoads != nullptr )
      {
          m_bodyLoads->get( mMesh, aState, aControl, aResult );
      }

      if( m_boundaryLoads != nullptr )
      {
          m_boundaryLoads->get( &mMesh, mMeshSets, aState, aControl, aResult );
      }

      if( std::count(m_plottable.begin(),m_plottable.end(),"strain") ) toMap(m_dataMap, tStrain, "strain");
      if( std::count(m_plottable.begin(),m_plottable.end(),"stress") ) toMap(m_dataMap, tStress, "stress");

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
