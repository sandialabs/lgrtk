#pragma once

#include "plato/AbstractLocalMeasure.hpp"
#include <Omega_h_matrix.hpp>
#include "plato/LinearStress.hpp"
#include "plato/Plato_VonMisesYield.hpp"
#include "plato/Strain.hpp"
#include "plato/ImplicitFunctors.hpp"
#include <Teuchos_ParameterList.hpp>
#include "plato/SimplexFadTypes.hpp"
#include "plato/LinearElasticMaterial.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief VonMises local measure class for use in Augmented Lagrange constraint formulation
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class VonMisesLocalMeasure :
        public AbstractLocalMeasure<EvaluationType>
{
private:
    using AbstractLocalMeasure<EvaluationType>::mSpaceDim; /*!< space dimension */
    using AbstractLocalMeasure<EvaluationType>::mNumVoigtTerms; /*!< number of voigt tensor terms */
    using AbstractLocalMeasure<EvaluationType>::mNumNodesPerCell; /*!< number of nodes per cell */
    Omega_h::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffMatrix; /*!< cell/element Lame constants matrix */

    using StateT = typename EvaluationType::StateScalarType; /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */

public:
    /******************************************************************************//**
     * @brief Primary constructor
     * @param [in] aInputParams input parameters database
     * @param [in] aName local measure name
     **********************************************************************************/
    VonMisesLocalMeasure(Teuchos::ParameterList & aInputParams,
                         const std::string & aName) : 
                         AbstractLocalMeasure<EvaluationType>(aInputParams, aName)
    {
        Plato::ElasticModelFactory<mSpaceDim> tMaterialModelFactory(aInputParams);
        auto tMaterialModel = tMaterialModelFactory.create();
        mCellStiffMatrix = tMaterialModel->getStiffnessMatrix();
    }

    /******************************************************************************//**
     * @brief Constructor tailored for unit testing
     * @param [in] aCellStiffMatrix stiffness matrix
     * @param [in] aName local measure name
     **********************************************************************************/
    VonMisesLocalMeasure(const Omega_h::Matrix<mNumVoigtTerms, mNumVoigtTerms> & aCellStiffMatrix,
                         const std::string aName) :
                         AbstractLocalMeasure<EvaluationType>(aName)
    {
        mCellStiffMatrix = aCellStiffMatrix;
    }

    /******************************************************************************//**
     * @brief Destructor
     **********************************************************************************/
    virtual ~VonMisesLocalMeasure()
    {
    }

    /******************************************************************************//**
     * @brief Evaluate vonmises local measure
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
        Plato::VonMisesYield<mSpaceDim> tComputeVonMises;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::LinearStress<mSpaceDim> tComputeCauchyStress(mCellStiffMatrix);

        // ****** ALLOCATE TEMPORARY MULTI-DIM ARRAYS ON DEVICE ******
        Plato::ScalarVectorT<ConfigT> tVolume("cell volume", tNumCells);
        Plato::ScalarMultiVectorT<StrainT> tCauchyStrain("strain", tNumCells, mNumVoigtTerms);
        Plato::ScalarMultiVectorT<ResultT> tCauchyStress("stress", tNumCells, mNumVoigtTerms);
        Plato::ScalarArray3DT<ConfigT> tGradient("gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
        {
            tComputeGradient(tCellOrdinal, tGradient, aConfigWS, tVolume);
            tComputeCauchyStrain(tCellOrdinal, tCauchyStrain, aStateWS, tGradient);
            tComputeCauchyStress(tCellOrdinal, tCauchyStress, tCauchyStrain);
            tComputeVonMises(tCellOrdinal, tCauchyStress, aResultWS);
        }, "Compute VonMises Stress");
    }
};
// class VonMisesLocalMeasure

}
//namespace Plato

#include "plato/SimplexMechanics.hpp"

#ifdef PLATO_1D
extern template class Plato::VonMisesLocalMeasure<Plato::ResidualTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::VonMisesLocalMeasure<Plato::JacobianTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::VonMisesLocalMeasure<Plato::GradientXTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::VonMisesLocalMeasure<Plato::GradientZTypes<Plato::SimplexMechanics<1>>>;
#endif

#ifdef PLATO_2D
extern template class Plato::VonMisesLocalMeasure<Plato::ResidualTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::VonMisesLocalMeasure<Plato::JacobianTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::VonMisesLocalMeasure<Plato::GradientXTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::VonMisesLocalMeasure<Plato::GradientZTypes<Plato::SimplexMechanics<2>>>;
#endif

#ifdef PLATO_3D
extern template class Plato::VonMisesLocalMeasure<Plato::ResidualTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::VonMisesLocalMeasure<Plato::JacobianTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::VonMisesLocalMeasure<Plato::GradientXTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::VonMisesLocalMeasure<Plato::GradientZTypes<Plato::SimplexMechanics<3>>>;
#endif