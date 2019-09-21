#ifndef STABILIZED_ELASTOSTATIC_ENERGY_HPP
#define STABILIZED_ELASTOSTATIC_ENERGY_HPP

#include "plato/SimplexFadTypes.hpp"
#include "plato/SimplexStabilizedMechanics.hpp"
#include "plato/ScalarProduct.hpp"
#include "plato/ProjectToNode.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/Kinematics.hpp"
#include "plato/Kinetics.hpp"
#include "plato/ImplicitFunctors.hpp"
#include "plato/InterpolateFromNodal.hpp"
#include "plato/AbstractScalarFunctionInc.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"
#include "plato/LinearElasticMaterial.hpp"
#include "plato/ToMap.hpp"
#include "plato/ExpInstMacros.hpp"

#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Compute internal elastic energy criterion for stabilized form.
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class StabilizedElastostaticEnergy : 
  public Plato::SimplexStabilizedMechanics<EvaluationType::SpatialDim>,
  public Plato::AbstractScalarFunctionInc<EvaluationType>
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;

    static constexpr Plato::OrdinalType mNMechDims  = mSpaceDim;
    static constexpr Plato::OrdinalType mNPressDims = 1;

    static constexpr Plato::OrdinalType mMDofOffset = 0;
    static constexpr Plato::OrdinalType mPDofOffset = mSpaceDim;
    
    using Plato::SimplexStabilizedMechanics<mSpaceDim>::mNumVoigtTerms;
    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell;
    using Plato::SimplexStabilizedMechanics<mSpaceDim>::mNumDofsPerNode;
    using Plato::SimplexStabilizedMechanics<mSpaceDim>::mNumDofsPerCell;

    using Plato::AbstractScalarFunctionInc<EvaluationType>::mMesh;
    using Plato::AbstractScalarFunctionInc<EvaluationType>::mDataMap;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using PrevStateScalarType = typename EvaluationType::PrevStateScalarType;
    using NodeStateScalarType = typename EvaluationType::NodeStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;

    
    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mSpaceDim, mNumVoigtTerms, IndicatorFunctionType> mApplyTensorWeighting;
    Plato::ApplyWeighting<mSpaceDim, mSpaceDim,      IndicatorFunctionType> mApplyVectorWeighting;
    Plato::ApplyWeighting<mSpaceDim, 1,              IndicatorFunctionType> mApplyScalarWeighting;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<mSpaceDim>> mCubatureRule;

    Teuchos::RCP<Plato::LinearElasticMaterial<mSpaceDim>> mMaterialModel;

    std::vector<std::string> mPlottable;

  public:
    /**************************************************************************/
    StabilizedElastostaticEnergy(Omega_h::Mesh&          aMesh,
                                 Omega_h::MeshSets&      aMeshSets,
                                 Plato::DataMap&         aDataMap,
                                 Teuchos::ParameterList& aProblemParams,
                                 Teuchos::ParameterList& aPenaltyParams,
                                 std::string&            aFunctionName ) :
            AbstractScalarFunctionInc<EvaluationType>(aMesh, aMeshSets, aDataMap, aFunctionName),
            mIndicatorFunction(aPenaltyParams),
            mApplyTensorWeighting(mIndicatorFunction),
            mApplyVectorWeighting(mIndicatorFunction),
            mApplyScalarWeighting(mIndicatorFunction),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<mSpaceDim>>())
    /**************************************************************************/
    {
      Plato::ElasticModelFactory<mSpaceDim> mmfactory(aProblemParams);
      mMaterialModel = mmfactory.create();

      if( aProblemParams.isType<Teuchos::Array<std::string>>("Plottable") )
        mPlottable = aProblemParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
    }

    /**************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT <StateScalarType>     & aStateWS,
                  const Plato::ScalarMultiVectorT <PrevStateScalarType> & aPrevStateWS,
                  const Plato::ScalarMultiVectorT <ControlScalarType>   & aControlWS,
                  const Plato::ScalarArray3DT     <ConfigScalarType>    & aConfigWS,
                        Plato::ScalarVectorT      <ResultScalarType>    & aResultWS,
                        Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      auto tNumCells = mMesh.nelems();

      using GradScalarType =
      typename Plato::fad_type_t<Plato::SimplexStabilizedMechanics
                <EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientWorkset <mSpaceDim> computeGradient;
      Plato::StabilizedKinematics   <mSpaceDim> kinematics;
      Plato::StabilizedKinetics     <mSpaceDim> kinetics(mMaterialModel);

      Plato::InterpolateFromNodal   <mSpaceDim, mSpaceDim, 0, mSpaceDim>     interpolatePGradFromNodal;
      Plato::InterpolateFromNodal   <mSpaceDim, mNumDofsPerNode, mPDofOffset> interpolatePressureFromNodal;

      Plato::ScalarProduct<mNumVoigtTerms> deviatorScalarProduct;
      
      Plato::ScalarVectorT      <ResultScalarType>    tVolStrain      ("volume strain",      tNumCells);
      Plato::ScalarVectorT      <ResultScalarType>    tPressure       ("GP pressure",        tNumCells);
      Plato::ScalarVectorT      <ConfigScalarType>    tCellVolume     ("cell weight",        tNumCells);
      Plato::ScalarMultiVectorT <NodeStateScalarType> tProjectedPGrad ("projected p grad",   tNumCells, mSpaceDim);
      Plato::ScalarMultiVectorT <ResultScalarType>    tCellStab       ("cell stabilization", tNumCells, mSpaceDim);
      Plato::ScalarMultiVectorT <GradScalarType>      tPGrad          ("pressure grad",      tNumCells, mSpaceDim);
      Plato::ScalarMultiVectorT <ResultScalarType>    tDevStress      ("deviatoric stress",  tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT <ResultScalarType>    tTotStress      ("cauchy stress",      tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT <GradScalarType>      tDGrad          ("displacement grad",  tNumCells, mNumVoigtTerms);
      Plato::ScalarArray3DT     <ConfigScalarType>    tGradient       ("gradient",           tNumCells, mNumNodesPerCell, mSpaceDim);

      auto tQuadratureWeight = mCubatureRule->getCubWeight();
      auto tBasisFunctions   = mCubatureRule->getBasisFunctions();

      auto& applyTensorWeighting = mApplyTensorWeighting;

      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(int cellOrdinal)
      {
        // compute gradient operator and cell volume
        //
        computeGradient(cellOrdinal, tGradient, aConfigWS, tCellVolume);
        tCellVolume(cellOrdinal) *= tQuadratureWeight;

        // compute symmetric gradient of displacement, pressure gradient, and temperature gradient
        //
        kinematics(cellOrdinal, tDGrad, tPGrad, aStateWS, tGradient);

        // interpolate projected PGrad, pressure, and temperature to gauss point
        //
        interpolatePressureFromNodal     ( cellOrdinal, tBasisFunctions, aStateWS, tPressure       );

        // compute the constitutive response
        //
        kinetics(cellOrdinal,     tCellVolume,
                 tProjectedPGrad, tPressure,
                 tDGrad,          tPGrad,
                 tDevStress,      tVolStrain,  tCellStab);

        for( int i=0; i<mSpaceDim; i++)
        {
            tTotStress(cellOrdinal,i) = tDevStress(cellOrdinal,i) + tPressure(cellOrdinal);
        }

        // apply weighting
        //
        applyTensorWeighting (cellOrdinal, tTotStress, aControlWS);

        // compute element internal energy (inner product of strain and weighted stress)
        //
        deviatorScalarProduct(cellOrdinal, aResultWS, tTotStress, tDGrad, tCellVolume);
      });
    }
};
// class InternalThermoelasticEnergy

} // namespace Plato

#ifdef PLATO_1D
PLATO_EXPL_DEC(Plato::StabilizedElastostaticEnergy, Plato::SimplexStabilizedMechanics, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEC(Plato::StabilizedElastostaticEnergy, Plato::SimplexStabilizedMechanics, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEC(Plato::StabilizedElastostaticEnergy, Plato::SimplexStabilizedMechanics, 3)
#endif

#endif
