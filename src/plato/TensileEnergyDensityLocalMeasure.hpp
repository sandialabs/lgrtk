#pragma once

#include "plato/AbstractLocalMeasure.hpp"
#include <Omega_h_matrix.hpp>
#include "plato/LinearStress.hpp"
#include "plato/TensileEnergyDensity.hpp"
#include "plato/Strain.hpp"
#include "plato/ImplicitFunctors.hpp"
#include <Teuchos_ParameterList.hpp>
#include "plato/SimplexFadTypes.hpp"
#include "plato/Eigenvalues.hpp"

namespace Plato
{
/******************************************************************************//**
 * @brief TensileEnergyDensity local measure class for use in Augmented Lagrange constraint formulation
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class TensileEnergyDensityLocalMeasure :
        public AbstractLocalMeasure<EvaluationType>
{
private:
    using AbstractLocalMeasure<EvaluationType>::mSpaceDim;
    using AbstractLocalMeasure<EvaluationType>::mNumVoigtTerms;
    using AbstractLocalMeasure<EvaluationType>::mNumNodesPerCell;
    Omega_h::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffMatrix; /*!< cell/element Lame constants matrix */

    using StateT = typename EvaluationType::StateScalarType; /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */

    Plato::Scalar mLameConstantLambda, mLameConstantMu, mPoissonsRatio, mYoungsModulus;

    /******************************************************************************//**
     * @brief Get Youngs Modulus and Poisson's Ratio from input parameter list
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void getYoungsModulusAndPoissonsRatio(Teuchos::ParameterList & aInputParams)
    {
        auto modelParamList = aInputParams.get<Teuchos::ParameterList>("Material Model");

        if( modelParamList.isSublist("Isotropic Linear Elastic") ){
            auto paramList = modelParamList.sublist("Isotropic Linear Elastic");
            mPoissonsRatio = paramList.get<double>("Poissons Ratio");
            mYoungsModulus = paramList.get<double>("Youngs Modulus");
        }
        else
        {
            throw std::runtime_error("Tensile Energy Density requires Isotropic Linear Elastic Material Model in ParameterList");
        }
    }

    /******************************************************************************//**
     * @brief Compute lame constants for isotropic linear elasticity
    **********************************************************************************/
    void computeLameConstants()
    {
        mLameConstantMu     = mYoungsModulus / 
                             (static_cast<Plato::Scalar>(2.0) * (static_cast<Plato::Scalar>(1.0) + 
                              mPoissonsRatio));
        mLameConstantLambda = static_cast<Plato::Scalar>(2.0) * mLameConstantMu * mPoissonsRatio / 
                             (static_cast<Plato::Scalar>(1.0) - static_cast<Plato::Scalar>(2.0) * mPoissonsRatio);
    }

public:
    /******************************************************************************//**
     * @brief Primary constructor
     * @param [in] aInputParams input parameters database
     * @param [in] aName local measure name
     **********************************************************************************/
    TensileEnergyDensityLocalMeasure(Teuchos::ParameterList & aInputParams,
                                     const std::string & aName) : 
                                     AbstractLocalMeasure<EvaluationType>(aInputParams, aName)
    {
        getYoungsModulusAndPoissonsRatio(aInputParams);
        computeLameConstants();
    }

    /******************************************************************************//**
     * @brief Constructor tailored for unit testing
     * @param [in] aYoungsModulus elastic modulus
     * @param [in] aPoissonsRatio Poisson's ratio
     * @param [in] aName local measure name
     **********************************************************************************/
    TensileEnergyDensityLocalMeasure(const Plato::Scalar & aYoungsModulus,
                                     const Plato::Scalar & aPoissonsRatio,
                                     const std::string & aName) :
                                     AbstractLocalMeasure<EvaluationType>(aName),
                                     mYoungsModulus(aYoungsModulus),
                                     mPoissonsRatio(aPoissonsRatio)
    {
        computeLameConstants();
    }

    /******************************************************************************//**
     * @brief Destructor
     **********************************************************************************/
    virtual ~TensileEnergyDensityLocalMeasure()
    {
    }

    /******************************************************************************//**
     * @brief Evaluate tensile energy density local measure
     * @param [in] aState 2D container of state variables
     * @param [in] aConfig 3D container of configuration/coordinates
     * @param [in] aDataMap map to stored data
     * @param [out] aResult 1D container of cell local measure values
    **********************************************************************************/
    virtual void operator()(const Plato::ScalarMultiVectorT<StateT> & aStateWS,
                            const Plato::ScalarArray3DT<ConfigT> & aConfigWS,
                            Plato::DataMap & aDataMap,
                            Plato::ScalarVectorT<ResultT> & aResultWS)
    {
        const Plato::OrdinalType tNumCells = aResultWS.size();
        using StrainT = typename Plato::fad_type_t<Plato::SimplexMechanics<mSpaceDim>, StateT, ConfigT>;

        Plato::Strain<mSpaceDim> tComputeCauchyStrain;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::Eigenvalues<mSpaceDim> tComputeEigenvalues;
        Plato::TensileEnergyDensity<mSpaceDim> tComputeTensileEnergyDensity;

        // ****** ALLOCATE TEMPORARY MULTI-DIM ARRAYS ON DEVICE ******
        Plato::ScalarVectorT<ConfigT> tVolume("cell volume", tNumCells);
        Plato::ScalarMultiVectorT<ResultT> tCauchyStrain("strain", tNumCells, mNumVoigtTerms);
        Plato::ScalarMultiVectorT<ResultT> tPrincipalStrains("principal strains", tNumCells, mSpaceDim);
        Plato::ScalarArray3DT<ConfigT> tGradient("gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

        const Plato::Scalar tLameLambda = mLameConstantLambda;
        const Plato::Scalar tLameMu     = mLameConstantMu;
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
        {
            tComputeGradient(tCellOrdinal, tGradient, aConfigWS, tVolume);
            tComputeCauchyStrain(tCellOrdinal, tCauchyStrain, aStateWS, tGradient);
            tComputeEigenvalues(tCellOrdinal, tCauchyStrain, tPrincipalStrains, true);
            tComputeTensileEnergyDensity(tCellOrdinal, tPrincipalStrains, tLameLambda, tLameMu, aResultWS);
        }, "Compute Tensile Energy Density");
    }
};
// class TensileEnergyDensityLocalMeasure

}
//namespace Plato

#include "plato/Mechanics.hpp"

#ifdef PLATO_1D
extern template class Plato::TensileEnergyDensityLocalMeasure<Plato::ResidualTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::TensileEnergyDensityLocalMeasure<Plato::JacobianTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::TensileEnergyDensityLocalMeasure<Plato::GradientXTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::TensileEnergyDensityLocalMeasure<Plato::GradientZTypes<Plato::SimplexMechanics<1>>>;
#endif

#ifdef PLATO_2D
extern template class Plato::TensileEnergyDensityLocalMeasure<Plato::ResidualTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::TensileEnergyDensityLocalMeasure<Plato::JacobianTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::TensileEnergyDensityLocalMeasure<Plato::GradientXTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::TensileEnergyDensityLocalMeasure<Plato::GradientZTypes<Plato::SimplexMechanics<2>>>;
#endif

#ifdef PLATO_3D
extern template class Plato::TensileEnergyDensityLocalMeasure<Plato::ResidualTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::TensileEnergyDensityLocalMeasure<Plato::JacobianTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::TensileEnergyDensityLocalMeasure<Plato::GradientXTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::TensileEnergyDensityLocalMeasure<Plato::GradientZTypes<Plato::SimplexMechanics<3>>>;
#endif