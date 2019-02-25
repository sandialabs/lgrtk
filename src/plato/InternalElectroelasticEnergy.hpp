#ifndef INTERNAL_ELECTROELASTIC_ENERGY_HPP
#define INTERNAL_ELECTROELASTIC_ENERGY_HPP

#include "plato/ScalarProduct.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/Strain.hpp"
#include "plato/LinearStress.hpp"
#include "plato/AbstractScalarFunction.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"
#include "plato/ToMap.hpp"

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class InternalElectroelasticEnergy : 
  public Plato::SimplexElectromechanics<EvaluationType::SpatialDim>,
  public AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;
    
    using Plato::SimplexElectromechanics<SpaceDim>::m_numVoigtTerms;
    using Simplex<SpaceDim>::m_numNodesPerCell;
    using Plato::SimplexElectromechanics<SpaceDim>::m_numDofsPerCell;

    using AbstractScalarFunction<EvaluationType>::mMesh;
    using AbstractScalarFunction<EvaluationType>::m_dataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Teuchos::RCP<Plato::LinearElectroelasticMaterial<SpaceDim>> m_materialModel;
    
    Plato::Scalar m_quadratureWeight;

    IndicatorFunctionType m_indicatorFunction;
    ApplyWeighting<SpaceDim, m_numVoigtTerms, IndicatorFunctionType> m_applyStressWeighting;
    ApplyWeighting<SpaceDim, SpaceDim,        IndicatorFunctionType> m_applyEDispWeighting;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> m_CubatureRule;

    std::vector<std::string> m_plottable;

  public:
    /**************************************************************************/
    InternalElectroelasticEnergy(Omega_h::Mesh& aMesh,
                          Omega_h::MeshSets& aMeshSets,
                          Plato::DataMap& aDataMap,
                          Teuchos::ParameterList& aProblemParams,
                          Teuchos::ParameterList& aPenaltyParams ) :
            AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Internal Electroelastic Energy"),
            m_indicatorFunction(aPenaltyParams),
            m_applyStressWeighting(m_indicatorFunction),
            m_applyEDispWeighting(m_indicatorFunction),
            m_CubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    /**************************************************************************/
    {
      Plato::ElectroelasticModelFactory<SpaceDim> mmfactory(aProblemParams);
      m_materialModel = mmfactory.create();

      if( aProblemParams.isType<Teuchos::Array<std::string>>("Plottable") )
        m_plottable = aProblemParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
    }

    /**************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> & aState,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarVectorT<ResultScalarType> & aResult,
                  Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      auto numCells = mMesh.nelems();

      using GradScalarType = 
        typename Plato::fad_type_t<Plato::SimplexElectromechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientWorkset<SpaceDim> computeGradient;
      EMKinematics<SpaceDim>                  kinematics;
      EMKinetics<SpaceDim>                    kinetics(m_materialModel);

      ScalarProduct<m_numVoigtTerms>          mechanicalScalarProduct;
      ScalarProduct<SpaceDim>                 electricalScalarProduct;

      Plato::ScalarVectorT<ConfigScalarType> cellVolume("cell weight",numCells);

      Plato::ScalarMultiVectorT<GradScalarType>   strain("strain", numCells, m_numVoigtTerms);
      Plato::ScalarMultiVectorT<GradScalarType>   efield("efield", numCells, SpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType> stress("stress", numCells, m_numVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> edisp ("edisp" , numCells, SpaceDim);

      Plato::ScalarArray3DT<ConfigScalarType>   gradient("gradient",numCells,m_numNodesPerCell,SpaceDim);

      auto quadratureWeight = m_CubatureRule->getCubWeight();

      auto& applyStressWeighting = m_applyStressWeighting;
      auto& applyEDispWeighting  = m_applyEDispWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
      {
        computeGradient(aCellOrdinal, gradient, aConfig, cellVolume);
        cellVolume(aCellOrdinal) *= quadratureWeight;

        // compute strain and electric field
        //
        kinematics(aCellOrdinal, strain, efield, aState, gradient);

        // compute stress and electric displacement
        //
        kinetics(aCellOrdinal, stress, edisp, strain, efield);

        // apply weighting
        //
        applyStressWeighting(aCellOrdinal, stress, aControl);
        applyEDispWeighting (aCellOrdinal, edisp,  aControl);
    
        // compute element internal energy (inner product of strain and weighted stress)
        //
        mechanicalScalarProduct(aCellOrdinal, aResult, stress, strain, cellVolume);
        electricalScalarProduct(aCellOrdinal, aResult, edisp,  efield, cellVolume);

      },"energy gradient");

      if( std::count(m_plottable.begin(),m_plottable.end(),"strain") ) toMap(m_dataMap, strain, "strain");
      if( std::count(m_plottable.begin(),m_plottable.end(),"stress") ) toMap(m_dataMap, stress, "stress");
      if( std::count(m_plottable.begin(),m_plottable.end(),"stress") ) toMap(m_dataMap, stress, "stress");
      if( std::count(m_plottable.begin(),m_plottable.end(),"edisp" ) ) toMap(m_dataMap, stress, "edisp" );

    }
};

#endif
