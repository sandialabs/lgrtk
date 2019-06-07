#ifndef VOLUME_HPP
#define VOLUME_HPP

#include "plato/ApplyWeighting.hpp"
#include "plato/Simplex.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/AbstractScalarFunction.hpp"
#include "plato/SimplexMechanics.hpp"
#include "plato/WorksetBase.hpp"
#include "plato/Plato_TopOptFunctors.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"
#include "ImplicitFunctors.hpp"
#include "plato/PlatoMathHelpers.hpp"

#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"
#include "plato/ExpInstMacros.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType, typename PenaltyFunctionType>
class Volume : public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
               public Plato::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int mSpaceDim = EvaluationType::SpatialDim;
    static constexpr Plato::OrdinalType mNumVoigtTerms = Plato::SimplexMechanics<mSpaceDim>::m_numVoigtTerms; /*!< number of Voigt terms */
    static constexpr Plato::OrdinalType mNumNodesPerCell = Plato::SimplexMechanics<mSpaceDim>::m_numNodesPerCell; /*!< number of nodes per cell/element */
    
    using Plato::AbstractScalarFunction<EvaluationType>::mMesh;
    using Plato::AbstractScalarFunction<EvaluationType>::m_dataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Plato::Scalar mCellMaterialDensity;
    Plato::Scalar mMassOfFullDesignVolume;

    PenaltyFunctionType mPenaltyFunction;
    Plato::ApplyWeighting<mSpaceDim,1,PenaltyFunctionType> mApplyWeighting;

  public:
    /**************************************************************************/
    Volume(Omega_h::Mesh& aMesh, 
           Omega_h::MeshSets& aMeshSets,
           Plato::DataMap& aDataMap, 
           Teuchos::ParameterList& aInputParams, 
           Teuchos::ParameterList& aPenaltyParams) :
            Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Volume"),
            mPenaltyFunction(aPenaltyParams),
            mApplyWeighting(mPenaltyFunction),
            mCellMaterialDensity(1.0),
            mMassOfFullDesignVolume(1.0)
    /**************************************************************************/
    {
      auto tMaterialModelInputs = aInputParams.get<Teuchos::ParameterList>("Material Model");
      mCellMaterialDensity = tMaterialModelInputs.get<Plato::Scalar>("Density", 1.0);

      computeMassOfFullDesignVolume ();
    }

    /**************************************************************************
     * Unit testing constructor
    /**************************************************************************/
    Volume(Omega_h::Mesh& aMesh, 
           Omega_h::MeshSets& aMeshSets,
           Plato::DataMap& aDataMap) :
            Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Volume"),
            mPenaltyFunction(3.0, 0.0),
            mApplyWeighting(mPenaltyFunction),
            mCellMaterialDensity(1.0),
            mMassOfFullDesignVolume(1.0)
    /**************************************************************************/
    {

    }

    /**************************************************************************/
    void setMaterialDensity(const Plato::Scalar aMaterialDensity)
    /**************************************************************************/
    {
      mCellMaterialDensity = aMaterialDensity;
      computeMassOfFullDesignVolume ();
    }

    /**************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> &,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarVectorT<ResultScalarType> & aResult,
                  Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      auto tNumCells = mMesh.nelems();

      Plato::ComputeCellVolume<mSpaceDim> tComputeCellVolume;

      auto tCellMaterialDensity  = mCellMaterialDensity;

      auto tMassOfFullDesignVolume = mMassOfFullDesignVolume;

      Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
      auto tCubWeight = tCubatureRule.getCubWeight();
      auto tBasisFunc = tCubatureRule.getBasisFunctions();

      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        ConfigScalarType tCellVolume;
        tComputeCellVolume(aCellOrdinal, aConfig, tCellVolume);
        tCellVolume *= tCubWeight;

        auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(aCellOrdinal, tBasisFunc, aControl);

        aResult(aCellOrdinal) = 
               ( tCellMass * tCellMaterialDensity * tCellVolume ) / tMassOfFullDesignVolume;

      },"volume fraction if one global material density");
    }

    /******************************************************************************//**
     * @brief Compute structural mass (i.e. structural mass with ersatz densities set to one)
    **********************************************************************************/
    void computeMassOfFullDesignVolume()
    {
      auto tNumCells = mMesh.nelems();
      Plato::NodeCoordinate<mSpaceDim> tCoordinates(&mMesh);
      Plato::ScalarArray3D tConfig("configuration", tNumCells, mNumNodesPerCell, mSpaceDim);
      Plato::workset_config_scalar<mSpaceDim, mNumNodesPerCell>(tNumCells, tCoordinates, tConfig);
      Plato::ComputeCellVolume<mSpaceDim> tComputeCellVolume;

      Plato::ScalarVector tTotalMass("total mass", tNumCells);
      Plato::ScalarMultiVector tDensities("densities", tNumCells, mNumNodesPerCell);
      Kokkos::deep_copy(tDensities, 1.0);

      Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
      auto tCellMaterialDensity = mCellMaterialDensity;
      auto tCubWeight = tCubatureRule.getCubWeight();
      auto tBasisFunc = tCubatureRule.getBasisFunctions();
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
          Plato::Scalar tCellVolume = 0;
          tComputeCellVolume(aCellOrdinal, tConfig, tCellVolume);
          tCellVolume *= tCubWeight;

          auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(aCellOrdinal, tBasisFunc, tDensities);

          tTotalMass(aCellOrdinal) = tCellMass * tCellMaterialDensity * tCellVolume;
          
      },"compute mass of full design volume");

      Plato::local_sum(tTotalMass, mMassOfFullDesignVolume);
    }
};
// class Volume

} // namespace Plato

#endif
