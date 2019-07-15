#pragma once

#include "plato/ScalarGrad.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/PlatoMathHelpers.hpp"
#include "plato/LinearTetCubRuleDegreeOne.hpp"
#include "plato/Simp.hpp"

#include "plato/AbstractLocalVectorFunctionInc.hpp"
#include "plato/ImplicitFunctors.hpp"
#include "plato/AnalyzeMacros.hpp"

#include "plato/J2PlasticityUtilities.hpp"
// Defined inside J2PlasticityUtilities.hpp
// #define SQRT_3_OVER_2 1.224744871391589
// #define SQRT_2_OVER_3 0.816496580927726
#include "plato/ThermoPlasticityUtilities.hpp"

#include "plato/ExpInstMacros.hpp"

namespace Plato
{

/**************************************************************************//**
* @brief J2 Plasticity Local Residual class
******************************************************************************/
template<typename EvaluationType, typename PhysicsType>
class J2PlasticityLocalResidual : 
  public Plato::AbstractLocalVectorFunctionInc<EvaluationType>
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

    using PhysicsType<mSpaceDim>::mNumNodesPerCell; /*!< number nodes per cell */
    using PhysicsType<mSpaceDim>::mNumVoigtTerms; /*!< number of voigt terms */

    using Plato::AbstractVectorFunctionInc<EvaluationType>::mMesh; /*!< mesh database */
    using Plato::AbstractVectorFunctionInc<EvaluationType>::mDataMap; /*!< PLATO Engine output database */

    using GlobalStateT     = typename EvaluationType::GlobalStateScalarType;     /*!< global state variables automatic differentiation type */
    using PrevGlobalStateT = typename EvaluationType::PrevGlobalStateScalarType; /*!< global state variables automatic differentiation type */
    using LocalStateT      = typename EvaluationType::LocalStateScalarType;      /*!< local state variables automatic differentiation type */
    using PrevLocalStateT  = typename EvaluationType::PrevLocalStateScalarType;  /*!< local state variables automatic differentiation type */
    using ControlT         = typename EvaluationType::ControlScalarType;         /*!< control variables automatic differentiation type */
    using ConfigT          = typename EvaluationType::ConfigScalarType;          /*!< config variables automatic differentiation type */
    using ResultT          = typename EvaluationType::ResultScalarType;          /*!< result variables automatic differentiation type */

    Plato::Scalar mElasticShearModulus;            /*!< elastic shear modulus */

    Plato::Scalar mThermalExpansionCoefficient;    /*!< thermal expansion coefficient */
    Plato::Scalar mReferenceTemperature;           /*!< reference temperature */

    Plato::Scalar mHardeningModulusIsotropic;      /*!< isotropic hardening modulus */
    Plato::Scalar mHardeningModulusKinematic;      /*!< kinematic hardening modulus */
    Plato::Scalar mInitialYieldStress;             /*!< initial yield stress */

    Plato::Scalar mElasticPropertiesPenaltySIMP;   /*!< SIMP penalty for elastic properties */
    Plato::Scalar mElasticPropertiesMinErsatzSIMP; /*!< SIMP min ersatz stiffness for elastic properties */

    Plato::Scalar mPlasticPropertiesPenaltySIMP;   /*!< SIMP penalty for plastic properties */
    Plato::Scalar mPlasticPropertiesMinErsatzSIMP; /*!< SIMP min ersatz stiffness for plastic properties */

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule; /*!< linear tet cubature rule */

    /**************************************************************************//**
    * @brief Return the names of the local state degrees of freedom
    * @return vector of local state dof names
    ******************************************************************************/
    std::vector<std::string> getLocalStateDofNames ()
    {
      if (mSpaceDim == 3)
      {
        std::vector<std::string> tDofNames(14);
        tDofNames[0]  = "Accumulated Plastic Strain";
        tDofNames[1]  = "Plastic Multiplier Increment";
        tDofNames[2]  = "Plastic Strain Tensor XX";
        tDofNames[3]  = "Plastic Strain Tensor YY";
        tDofNames[4]  = "Plastic Strain Tensor ZZ";
        tDofNames[5]  = "Plastic Strain Tensor YZ";
        tDofNames[6]  = "Plastic Strain Tensor XZ";
        tDofNames[7]  = "Plastic Strain Tensor XY";
        tDofNames[8]  = "Backstress Tensor XX";
        tDofNames[9]  = "Backstress Tensor YY";
        tDofNames[10] = "Backstress Tensor ZZ";
        tDofNames[11] = "Backstress Tensor YZ";
        tDofNames[12] = "Backstress Tensor XZ";
        tDofNames[13] = "Backstress Tensor XY";
        return tDofNames;
      }
      else if (mSpaceDim == 2)
      {
        std::vector<std::string> tDofNames(8);
        tDofNames[0] = "Accumulated Plastic Strain";
        tDofNames[1] = "Plastic Multiplier Increment";
        tDofNames[2] = "Plastic Strain Tensor XX";
        tDofNames[3] = "Plastic Strain Tensor YY";
        tDofNames[4] = "Plastic Strain Tensor XY";
        tDofNames[5] = "Backstress Tensor XX";
        tDofNames[6] = "Backstress Tensor YY";
        tDofNames[7] = "Backstress Tensor XY";
        return tDofNames;
      }
      else
      {
        THROWERR("J2 Plasticity Local Residual not implemented for space dim other than 2 or 3.")
      }
    }

    /**************************************************************************//**
    * @brief Initialize problem parameters
    * @param [in] aProblemParams Teuchos parameter list
    ******************************************************************************/
    void initialize(Teuchos::ParameterList& aProblemParams)
    {
      auto tMaterialParamList = aProblemParams.get<Teuchos::ParameterList>("Material Model");

      if( tMaterialParamList.isSublist("Isotropic Linear Elastic") )
      {
        auto tElasticSubList = tMaterialParamList.sublist("Isotropic Linear Elastic");
        mThermalExpansionCoefficient = 0.0;
        mReferenceTemperature        = 0.0;

        auto tElasticModulus = tElasticSubList.get<double>("Youngs Modulus");
        auto tPoissonsRatio  = tElasticSubList.get<double>("Poissons Ratio");
        mElasticShearModulus = tElasticModulus / (2.0 * (1.0 + tPoissonsRatio));
      }
      else if( tMaterialParamList.isSublist("Isotropic Linear Thermoelastic") )
      {
        auto tThermoelasticSubList = tMaterialParamList.sublist("Isotropic Linear Thermoelastic");

        mThermalExpansionCoefficient = tThermoelasticSubList.get<double>("Thermal Expansion Coefficient");
        mReferenceTemperature        = tThermoelasticSubList.get<double>("Reference Temperature");

        auto tElasticModulus = tThermoelasticSubList.get<double>("Youngs Modulus");
        auto tPoissonsRatio  = tThermoelasticSubList.get<double>("Poissons Ratio");
        mElasticShearModulus = tElasticModulus / (2.0 * (1.0 + tPoissonsRatio));
      }
      else
      {
        THROWERR("'Isotropic Linear Elastic' or 'Isotropic Linear Thermoelastic' sublist of 'Material Model' does not exist.")
      }

      if( tMaterialParamList.isSublist("J2 Plasticity") )
      {
        auto tJ2PlasticitySubList = tMaterialParamList.sublist("J2 Plasticity");

        mHardeningModulusIsotropic = tJ2PlasticitySubList.get<double>("Hardening Modulus Isotropic");
        mHardeningModulusKinematic = tJ2PlasticitySubList.get<double>("Hardening Modulus Kinematic");
        mInitialYieldStress        = tJ2PlasticitySubList.get<double>("Initial Yield Stress");

        mElasticPropertiesPenaltySIMP   = tJ2PlasticitySubList.get<double>("Elastic Properties Penalty Exponent");
        mElasticPropertiesMinErsatzSIMP = tJ2PlasticitySubList.get<double>("Elastic Properties Minimum Ersatz");

        mPlasticPropertiesPenaltySIMP   = tJ2PlasticitySubList.get<double>("Plastic Properties Penalty Exponent");
        mPlasticPropertiesMinErsatzSIMP = tJ2PlasticitySubList.get<double>("Plastic Properties Minimum Ersatz");
      }
      else
      {
        THROWERR("'J2 Plasticity' sublist of 'Material Model' does not exist. Needed for J2Plasticity Implementation.")
      }
    }

  public:
    /**************************************************************************//**
    * @brief Constructor
    * @param [in] aMesh mesh data base
    * @param [in] aMeshSets mesh sets data base
    * @param [in] aDataMap problem-specific data map 
    * @param [in] aProblemParams Teuchos parameter list
    ******************************************************************************/
    J2PlasticityLocalResidual( Omega_h::Mesh& aMesh,
                               Omega_h::MeshSets& aMeshSets,
                               Plato::DataMap& aDataMap,
                               Teuchos::ParameterList& aProblemParams) :
     AbstractLocalVectorFunctionInc<EvaluationType>(aMesh, aMeshSets, aDataMap, getLocalStateDofNames() ),
     mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    {
      initialize(aProblemParams);
    }


    /**************************************************************************//**
    * @brief Evaluate the local J2 plasticity residual
    * @param [in] aGlobalState global state at current time step
    * @param [in] aPrevGlobalState global state at previous time step
    * @param [in] aLocalState local state at current time step
    * @param [in] aPrevLocalState local state at previous time step
    * @param [in] aControl control parameters
    * @param [in] aConfig configuration parameters
    * @param [out] aResult evaluated local residuals
    ******************************************************************************/
    void
    evaluate( const Plato::ScalarMultiVectorT< GlobalStateT >     & aGlobalState,
              const Plato::ScalarMultiVectorT< PrevGlobalStateT > & aPrevGlobalState,
              const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
              const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
              const Plato::ScalarMultiVectorT< ControlT >         & aControl,
              const Plato::ScalarArray3DT    < ConfigT >          & aConfig,
                    Plato::ScalarMultiVectorT< ResultT >          & aResult,
                    Plato::Scalar aTimeStep = 0.0) const
    {
      auto tNumCells = mMesh.nelems();

      using TotalStrainT =
          typename Plato::fad_type_t<PhysicsType<EvaluationType::SpatialDim>, GlobalStateT, ConfigT>;

      using ElasticStrainT =
          typename Plato::fad_type_t<PhysicsType<EvaluationType::SpatialDim>, GlobalStateT, LocalStateT>;

      using StressT =
          typename Plato::fad_type_t<PhysicsType<EvaluationType::SpatialDim>, ElasticStrainT, ConfigT>;

      // Functors
      Plato::ComputeGradientWorkset<EvaluationType::SpatialDim> tComputeGradient;

      // J2 Utility Functions Object
      Plato::J2PlasticityUtilities<EvaluationType::SpatialDim>  tJ2PlasticityUtils;

      // ThermoPlasticity Utility Functions Object (for computing elastic strain and potentially temperature-dependent material properties)
      Plato::ThermoPlasticityUtilities<EvaluationType::SpatialDim, PhysicsType> 
            tThermoPlasticityUtils(mThermalExpansionCoefficient, mReferenceTemperature);

      // Many views
      Plato::ScalarVectorT<ConfigT>             tCellVolume("cell volume unused", tNumCells);
      Plato::ScalarVectorT<StressT>             tDevStressMinusBackstressNorm("norm(deviatoric_stress - backstress)",tNumCells);
      Plato::ScalarMultiVectorT<ElasticStrainT> tElasticStrain("elastic strain", tNumCells,mNumVoigtTerms);
      Plato::ScalarMultiVectorT<StressT>        tDeviatoricStress("deviatoric stress", tNumCells,mNumVoigtTerms);
      Plato::ScalarMultiVectorT<StressT>        tYieldSurfaceNormal("yield surface normal",tNumCells,mNumVoigtTerms);
      Plato::ScalarArray3DT<ConfigT>            tGradient("gradient", tNumCells,mNumNodesPerCell,mSpaceDim);

      // Transfer elasticity parameters to device
      auto tElasticShearModulus = mElasticShearModulus;

      // Transfer plasticity parameters to device
      auto tHardeningModulusIsotropic = mHardeningModulusIsotropic;
      auto tHardeningModulusKinematic = mHardeningModulusKinematic;
      auto tInitialYieldStress        = mInitialYieldStress;

      auto tBasisFunctions = mCubatureRule->getBasisFunctions();

      Plato::MSIMP tElasticPropertiesSIMP(mElasticPropertiesPenaltySIMP, mElasticPropertiesMinErsatzSIMP);
      Plato::MSIMP tPlasticPropertiesSIMP(mPlasticPropertiesPenaltySIMP, mPlasticPropertiesMinErsatzSIMP);

      Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);

        // compute elastic strain
        tThermoPlasticityUtils.computeElasticStrain(aCellOrdinal, aGlobalState, aLocalState, tBasisFunctions, tGradient, tElasticStrain);
      
        // apply penalization to elastic shear modulus
        ControlT tDensity               = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, aControl);
        ControlT tElasticParamsPenalty  = tElasticPropertiesSIMP(tDensity);
        ControlT tPenalizedShearModulus = tElasticParamsPenalty * tElasticShearModulus;

        // compute deviatoric stress
        tJ2PlasticityUtils.computeDeviatoricStress(aCellOrdinal, tElasticStrain, tPenalizedShearModulus, tDeviatoricStress);

        // compute eta = (deviatoric_stress - backstress) ... and its norm ... the normalized version is the yield surface normal
        tJ2PlasticityUtils.computeDeviatoricStressMinusBackstressNormalized(aCellOrdinal, tDeviatoricStress, aLocalState,
                                                                            tYieldSurfaceNormal, tDevStressMinusBackstressNorm);

        // apply penalization to plasticity material parameters
        ControlT tPlasticParamsPenalty               = tPlasticPropertiesSIMP(tDensity);
        ControlT tPenalizedHardeningModulusIsotropic = tPlasticParamsPenalty * tHardeningModulusIsotropic;
        ControlT tPenalizedHardeningModulusKinematic = tPlasticParamsPenalty * tHardeningModulusKinematic;
        ControlT tPenalizedInitialYieldStress        = tPlasticParamsPenalty * tInitialYieldStress;

        // compute yield stress
        ResultT tYieldStress = tPenalizedInitialYieldStress + 
                               tPenalizedHardeningModulusIsotropic * aLocalState(aCellOrdinal, 0);

        if (aLocalState(aCellOrdinal, 1) /*Current Plastic Multiplier Increment*/ > 0.0) // -> yielding (assumes local state already updated)
        {
          // Residual: Accumulated Plastic Strain, DOF: Accumulated Plastic Strain
          aResult(aCellOrdinal, 0) = aLocalState(aCellOrdinal, 0) - aPrevLocalState(aCellOrdinal, 0)
                                                                  -     aLocalState(aCellOrdinal, 1);

          // Residual: Yield Function , DOF: Plastic Multiplier Increment
          aResult(aCellOrdinal, 1) = SQRT_3_OVER_2 * tDevStressMinusBackstressNorm(aCellOrdinal) - tYieldStress;

          // Residual: Plastic Strain Tensor, DOF: Plastic Strain Tensor
          tJ2PlasticityUtils.fillPlasticStrainTensorResidualPlasticStep(aCellOrdinal, aLocalState, aPrevLocalState,
                                                                        tYieldSurfaceNormal, aResult);

          // Residual: Backstress, DOF: Backstress
          tJ2PlasticityUtils.fillBackstressTensorResidualPlasticStep(aCellOrdinal,
                                                                     tPenalizedHardeningModulusKinematic,
                                                                     aLocalState,         aPrevLocalState,
                                                                     tYieldSurfaceNormal, aResult);
        }
        else // -> elastic step
        {
          // Residual: Accumulated Plastic Strain, DOF: Accumulated Plastic Strain
          aResult(aCellOrdinal, 0) = aLocalState(aCellOrdinal, 0) - aPrevLocalState(aCellOrdinal, 0);

          // Residual: Plastic Multiplier Increment = 0 , DOF: Plastic Multiplier Increment
          aResult(aCellOrdinal, 1) = aLocalState(aCellOrdinal, 1);

          // Residual: Plastic Strain Tensor, DOF: Plastic Strain Tensor
          tJ2PlasticityUtils.fillPlasticStrainTensorResidualElasticStep(aCellOrdinal, aLocalState, aPrevLocalState, aResult);

          // Residual: Backstress, DOF: Backstress
          tJ2PlasticityUtils.fillBackstressTensorResidualElasticStep(aCellOrdinal, aLocalState, aPrevLocalState, aResult);
        }

      }, "Compute cell local residuals");
    }

    /**************************************************************************//**
    * @brief Update the local state variables
    * @param [in]  aGlobalState global state at current time step
    * @param [in]  aPrevGlobalState global state at previous time step
    * @param [out] aLocalState local state at current time step
    * @param [in]  aPrevLocalState local state at previous time step
    * @param [in]  aControl control parameters
    * @param [in]  aConfig configuration parameters
    ******************************************************************************/
    void
    updateLocalState( const Plato::ScalarMultiVector & aGlobalState,
                      const Plato::ScalarMultiVector & aPrevGlobalState,
                      const Plato::ScalarMultiVector & aLocalState,
                      const Plato::ScalarMultiVector & aPrevLocalState,
                      const Plato::ScalarMultiVector & aControl,
                      const Plato::ScalarArray3D     & aConfig) const
    {
      auto tNumCells = mMesh.nelems();

      // Functors
      Plato::ComputeGradientWorkset<EvaluationType::SpatialDim> tComputeGradient;

      // J2 Utility Functions Object
      Plato::J2PlasticityUtilities<EvaluationType::SpatialDim>  tJ2PlasticityUtils;

      // ThermoPlasticity Utility Functions Object (for computing elastic strain and potentially temperature-dependent material properties)
      Plato::ThermoPlasticityUtilities<EvaluationType::SpatialDim, PhysicsType> 
            tThermoPlasticityUtils(mThermalExpansionCoefficient, mReferenceTemperature);

      // Many views
      Plato::ScalarVector      tCellVolume("cell volume unused",tNumCells);
      Plato::ScalarMultiVector tElasticStrain("elastic strain",tNumCells,mNumVoigtTerms);
      Plato::ScalarArray3D     tGradient("gradient",tNumCells,mNumNodesPerCell,mSpaceDim);
      Plato::ScalarMultiVector tDeviatoricStress("deviatoric stress",tNumCells,mNumVoigtTerms);
      Plato::ScalarMultiVector tYieldSurfaceNormal("yield surface normal",tNumCells,mNumVoigtTerms);
      Plato::ScalarVector      tDevStressMinusBackstressNorm("||(deviatoric stress - backstress)||",tNumCells);

      // Transfer elasticity parameters to device
      auto tElasticShearModulus = mElasticShearModulus;

      // Transfer plasticity parameters to device
      auto tHardeningModulusIsotropic = mHardeningModulusIsotropic;
      auto tHardeningModulusKinematic = mHardeningModulusKinematic;
      auto tInitialYieldStress        = mInitialYieldStress;

      auto tBasisFunctions = mCubatureRule->getBasisFunctions();

      Plato::MSIMP tElasticPropertiesSIMP(mElasticPropertiesPenaltySIMP, mElasticPropertiesMinErsatzSIMP);
      Plato::MSIMP tPlasticPropertiesSIMP(mPlasticPropertiesPenaltySIMP, mPlasticPropertiesMinErsatzSIMP);

      Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);

        // compute elastic strain
        tThermoPlasticityUtils.computeElasticStrain(aCellOrdinal, aGlobalState, aLocalState, tBasisFunctions, tGradient, tElasticStrain);
      
        // apply penalization to elastic shear modulus
        Plato::Scalar tDensity               = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, aControl);
        Plato::Scalar tElasticParamsPenalty  = tElasticPropertiesSIMP(tDensity);
        Plato::Scalar tPenalizedShearModulus = tElasticParamsPenalty * tElasticShearModulus;

        // compute deviatoric stress
        tJ2PlasticityUtils.computeDeviatoricStress(aCellOrdinal, tElasticStrain, tPenalizedShearModulus, tDeviatoricStress);

        // compute eta = (deviatoric_stress - backstress) ... and its norm ... the normalized version is the yield surf normal
        tJ2PlasticityUtils.computeDeviatoricStressMinusBackstressNormalized(aCellOrdinal, tDeviatoricStress, aLocalState,
                                                                            tYieldSurfaceNormal, tDevStressMinusBackstressNorm);

        // apply penalization to plasticity material parameters
        Plato::Scalar tPlasticParamsPenalty               = tPlasticPropertiesSIMP(tDensity);
        Plato::Scalar tPenalizedHardeningModulusIsotropic = tPlasticParamsPenalty * tHardeningModulusIsotropic;
        Plato::Scalar tPenalizedHardeningModulusKinematic = tPlasticParamsPenalty * tHardeningModulusKinematic;
        Plato::Scalar tPenalizedInitialYieldStress        = tPlasticParamsPenalty * tInitialYieldStress;

        // compute yield stress
        Plato::Scalar tYieldStress = tPenalizedInitialYieldStress + 
                                     tPenalizedHardeningModulusIsotropic * aLocalState(aCellOrdinal, 0);

        // compute the yield function at the trial state
        Plato::Scalar tTrialStateYieldFunction = SQRT_3_OVER_2 * tDevStressMinusBackstressNorm(aCellOrdinal) - tYieldStress;

        if (tTrialStateYieldFunction <= 0.0) // elastic step
        {
          // Accumulated Plastic Strain
          aLocalState(aCellOrdinal, 0) = aPrevLocalState(aCellOrdinal, 0);

          // Plastic Multiplier Increment
          aLocalState(aCellOrdinal, 1) = 0.0;
          
          tJ2PlasticityUtils.updatePlasticStrainAndBackstressElasticStep(aCellOrdinal, aPrevLocalState, aLocalState);
        }
        else // plastic step
        {
          // Plastic Multiplier Increment (for J2 w/ linear isotropic/kinematic hardening -> analytical return mapping)
          aLocalState(aCellOrdinal, 1) = tTrialStateYieldFunction / (3.0 * tPenalizedShearModulus +
                                         tPenalizedHardeningModulusIsotropic + tPenalizedHardeningModulusKinematic);

          // Accumulated Plastic Strain
          aLocalState(aCellOrdinal, 0) = aPrevLocalState(aCellOrdinal, 0) + aLocalState(aCellOrdinal, 1);

          tJ2PlasticityUtils.updatePlasticStrainAndBackstressPlasticStep(aCellOrdinal, aPrevLocalState, tYieldSurfaceNormal,
                                                                         tPenalizedHardeningModulusKinematic, aLocalState);
        }
      }, "Update local state dofs");
    }
};
// class J2PlasticityLocalResidual

} // namespace Plato

#ifdef PLATO_2D
PLATO_EXPL_DEC_INC(Plato::J2PlasticityLocalResidual, Plato::SimplexPlasticity, 2)
PLATO_EXPL_DEC_INC(Plato::J2PlasticityLocalResidual, Plato::SimplexThermoPlasticity, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEC_INC(Plato::J2PlasticityLocalResidual, Plato::SimplexPlasticity, 3)
PLATO_EXPL_DEC_INC(Plato::J2PlasticityLocalResidual, Plato::SimplexThermoPlasticity, 3)
#endif
