#ifndef EFFECTIVE_ELASTIC_ENERGY_HPP
#define EFFECTIVE_ELASTIC_ENERGY_HPP

#include "plato/ScalarProduct.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/Strain.hpp"
#include "plato/HomogenizedStress.hpp"
#include "plato/AbstractScalarFunction.hpp"

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class EffectiveEnergy : 
  public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
  public AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;
    
    using Plato::SimplexMechanics<SpaceDim>::m_numVoigtTerms;
    using Simplex<SpaceDim>::m_numNodesPerCell;
    using Plato::SimplexMechanics<SpaceDim>::m_numDofsPerCell;

    using AbstractScalarFunction<EvaluationType>::mMesh;
    using AbstractScalarFunction<EvaluationType>::m_dataMap;
    using AbstractScalarFunction<EvaluationType>::m_functionName;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType m_indicatorFunction;
    ApplyWeighting<SpaceDim,m_numVoigtTerms,IndicatorFunctionType> m_applyWeighting;

    Omega_h::Matrix< m_numVoigtTerms, m_numVoigtTerms> m_cellStiffness;
    Omega_h::Vector<m_numVoigtTerms> m_assumedStrain;
    int m_columnIndex;
    Plato::Scalar m_quadratureWeight;

  public:
    /**************************************************************************/
    EffectiveEnergy(Omega_h::Mesh& aMesh,
                    Omega_h::MeshSets& aMeshSets,
                    Plato::DataMap& aDataMap,
                    Teuchos::ParameterList& aProblemParams,
                    Teuchos::ParameterList& aPenaltyParams) :
            AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Effective Energy"),
            m_indicatorFunction(aPenaltyParams),
            m_applyWeighting(m_indicatorFunction)
    /**************************************************************************/
    {
      Plato::ElasticModelFactory<SpaceDim> mmfactory(aProblemParams);
      auto materialModel = mmfactory.create();
      m_cellStiffness = materialModel->getStiffnessMatrix();

      Teuchos::ParameterList& tParams = aProblemParams.sublist(m_functionName);
      auto tAssumedStrain = tParams.get<Teuchos::Array<double>>("Assumed Strain");
      assert(tAssumedStrain.size() == m_numVoigtTerms);
      for( int iVoigt=0; iVoigt<m_numVoigtTerms; iVoigt++)
      {
          m_assumedStrain[iVoigt] = tAssumedStrain[iVoigt];
      }

      // parse cell problem forcing
      //
      if(aProblemParams.isSublist("Cell Problem Forcing"))
      {
          m_columnIndex = aProblemParams.sublist("Cell Problem Forcing").get<int>("Column Index");
      }
      else
      {
          // JR TODO: throw
      }

      m_quadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
      for (int d=2; d<=SpaceDim; d++)
      { 
          m_quadratureWeight /= Plato::Scalar(d);
      }
    
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
      HomogenizedStress<SpaceDim>             homogenizedStress(m_cellStiffness, m_columnIndex);
      ScalarProduct<m_numVoigtTerms>          scalarProduct;

      using StrainScalarType = 
        typename Plato::fad_type_t<Plato::SimplexMechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        cellVolume("cell weight",numCells);

      Kokkos::View<StrainScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        strain("strain",numCells,m_numVoigtTerms);

      Kokkos::View<ConfigScalarType***, Kokkos::LayoutRight, Plato::MemSpace>
        gradient("gradient",numCells,m_numNodesPerCell,SpaceDim);

      Kokkos::View<ResultScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        stress("stress",numCells,m_numVoigtTerms);

      auto quadratureWeight = m_quadratureWeight;
      auto applyWeighting   = m_applyWeighting;
      auto assumedStrain    = m_assumedStrain;
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
      {
        computeGradient(aCellOrdinal, gradient, aConfig, cellVolume);
        cellVolume(aCellOrdinal) *= quadratureWeight;

        // compute strain
        //
        voigtStrain(aCellOrdinal, strain, aState, gradient);

        // compute stress
        //
        homogenizedStress(aCellOrdinal, stress, strain);

        // apply weighting
        //
        applyWeighting(aCellOrdinal, stress, aControl);
    
        // compute element internal energy (inner product of strain and weighted stress)
        //
        scalarProduct(aCellOrdinal, aResult, stress, assumedStrain, cellVolume);

      },"energy gradient");
    }
};

#endif
