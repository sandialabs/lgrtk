#ifndef STRESS_P_NORM_HPP
#define STRESS_P_NORM_HPP

#include "plato/ScalarProduct.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/Strain.hpp"
#include "plato/LinearStress.hpp"
#include "plato/TensorPNorm.hpp"
#include "plato/AbstractScalarFunction.hpp"

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class StressPNorm : 
  public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
  public AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType SpaceDim = EvaluationType::SpatialDim;
    
    using Plato::SimplexMechanics<SpaceDim>::m_numVoigtTerms;
    using Simplex<SpaceDim>::m_numNodesPerCell;
    using Plato::SimplexMechanics<SpaceDim>::m_numDofsPerCell;

    using AbstractScalarFunction<EvaluationType>::mMesh;
    using AbstractScalarFunction<EvaluationType>::m_dataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Omega_h::Matrix< m_numVoigtTerms, m_numVoigtTerms> m_cellStiffness;
    
    Plato::Scalar m_quadratureWeight;

    IndicatorFunctionType m_indicatorFunction;
    ApplyWeighting<SpaceDim,m_numVoigtTerms,IndicatorFunctionType> m_applyWeighting;

    Teuchos::RCP<TensorNormBase<m_numVoigtTerms,EvaluationType>> m_norm;

  public:
    /**************************************************************************/
    StressPNorm(Omega_h::Mesh& aMesh,
                Omega_h::MeshSets& aMeshSets,
                Plato::DataMap& aDataMap, 
                Teuchos::ParameterList& aProblemParams, 
                Teuchos::ParameterList& aPenaltyParams) :
            AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Stress P-Norm"),
            m_indicatorFunction(aPenaltyParams),
            m_applyWeighting(m_indicatorFunction)
    /**************************************************************************/
    {
      Plato::ElasticModelFactory<SpaceDim> mmfactory(aProblemParams);
      auto materialModel = mmfactory.create();
      m_cellStiffness = materialModel->getStiffnessMatrix();

      m_quadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
      for (Plato::OrdinalType d=2; d<=SpaceDim; d++)
      { 
        m_quadratureWeight /= Plato::Scalar(d);
      }

      auto params = aProblemParams.get<Teuchos::ParameterList>("Stress P-Norm");

      TensorNormFactory<m_numVoigtTerms, EvaluationType> normFactory;
      m_norm = normFactory.create(params);
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

      Plato::ComputeGradientWorkset<SpaceDim> computeGradient;
      Strain<SpaceDim>                        voigtStrain;
      LinearStress<SpaceDim>                  voigtStress(m_cellStiffness);

      using StrainScalarType = 
        typename Plato::fad_type_t<Plato::SimplexMechanics<EvaluationType::SpatialDim>,
                            StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        cellVolume("cell weight",numCells);

      Plato::ScalarMultiVectorT<StrainScalarType>
        strain("strain",numCells,m_numVoigtTerms);

      Plato::ScalarArray3DT<ConfigScalarType>
        gradient("gradient",numCells,m_numNodesPerCell,SpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType>
        stress("stress",numCells,m_numVoigtTerms);

      auto quadratureWeight = m_quadratureWeight;
      auto applyWeighting  = m_applyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,numCells), LAMBDA_EXPRESSION(Plato::OrdinalType cellOrdinal)
      {
        computeGradient(cellOrdinal, gradient, aConfig, cellVolume);
        cellVolume(cellOrdinal) *= quadratureWeight;

        // compute strain
        //
        voigtStrain(cellOrdinal, strain, aState, gradient);

        // compute stress
        //
        voigtStress(cellOrdinal, stress, strain);
      
        // apply weighting
        //
        applyWeighting(cellOrdinal, stress, aControl);

      },"Compute Stress");

      m_norm->evaluate(aResult, stress, aControl, cellVolume);

    }

    /**************************************************************************/
    void
    postEvaluate( 
      Plato::ScalarVector resultVector,
      Plato::Scalar       resultScalar)
    /**************************************************************************/
    {
      m_norm->postEvaluate(resultVector, resultScalar);
    }

    /**************************************************************************/
    void
    postEvaluate( Plato::Scalar& resultValue )
    /**************************************************************************/
    {
      m_norm->postEvaluate(resultValue);
    }
};

#endif
