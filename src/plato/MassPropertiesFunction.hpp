#pragma once

#include <memory>
#include <cassert>
#include <vector>

#include <Omega_h_mesh.hpp>

#include "plato/WorksetBase.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/ScalarFunctionBaseFactory.hpp"
#include "plato/PhysicsScalarFunction.hpp"
#include "plato/DivisionFunction.hpp"
#include "plato/LeastSquaresFunction.hpp"
#include "plato/WeightedSumFunction.hpp"
#include "plato/MassMoment.hpp"

#include <Teuchos_ParameterList.hpp>

namespace Plato
{

/******************************************************************************//**
 * @brief Mass properties function class
 **********************************************************************************/
template<typename PhysicsT>
class MassPropertiesFunction : public Plato::ScalarFunctionBase, public Plato::WorksetBase<PhysicsT>
{
private:
    using Residual = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Residual; /*!< result variables automatic differentiation type */
    using Jacobian = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Jacobian; /*!< state variables automatic differentiation type */
    using GradientX = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientX; /*!< configuration variables automatic differentiation type */
    using GradientZ = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientZ; /*!< control variables automatic differentiation type */

    std::shared_ptr<Plato::LeastSquaresFunction<PhysicsT>> mLeastSquaresFunction; /*!< Least squares function object */

    Plato::DataMap& m_dataMap; /*!< PLATO Engine and Analyze data map */

    std::string mFunctionName; /*!< User defined function name */

    Plato::Scalar mMaterialDensity;

	/******************************************************************************//**
     * @brief Initialization of Mass Properties Function
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize (Omega_h::Mesh& aMesh, 
                     Omega_h::MeshSets& aMeshSets, 
                     Teuchos::ParameterList & aInputParams)
    {
        auto tMaterialModelInputs = aInputParams.get<Teuchos::ParameterList>("Material Model");
        mMaterialDensity = tMaterialModelInputs.get<Plato::Scalar>("Density", 1.0);

        createLeastSquaresFunction(aMesh, aMeshSets, aInputParams);
    }

    /******************************************************************************//**
     * @brief Create the least squares mass properties function
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void createLeastSquaresFunction(Omega_h::Mesh& aMesh, 
                                    Omega_h::MeshSets& aMeshSets,
                                    Teuchos::ParameterList & aInputParams)
    {
        auto tProblemFunctionName = aInputParams.sublist(mFunctionName);

        auto tPropertyNamesTeuchos = tProblemFunctionName.get<Teuchos::Array<std::string>>("Properties");
        auto tPropertyWeightsTeuchos = tProblemFunctionName.get<Teuchos::Array<double>>("Weights");
        auto tPropertyGoldValuesTeuchos = tProblemFunctionName.get<Teuchos::Array<double>>("Gold Values");

        auto tPropertyNames      = tPropertyNamesTeuchos.toVector();
        auto tPropertyWeights    = tPropertyWeightsTeuchos.toVector();
        auto tPropertyGoldValues = tPropertyGoldValuesTeuchos.toVector();

        if (tPropertyNames.size() != tPropertyWeights.size())
        {
            const std::string tErrorString = std::string("Number of 'Properties' in '") + mFunctionName + 
                                                         "' parameter list does not equal the number of 'Weights'";
            throw std::runtime_error(tErrorString);
        }

        if (tPropertyNames.size() != tPropertyGoldValues.size())
        {
            const std::string tErrorString = std::string("Number of 'Gold Values' in '") + mFunctionName + 
                                                         "' parameter list does not equal the number of 'Properties'";
            throw std::runtime_error(tErrorString);
        }

        mLeastSquaresFunction = std::make_shared<Plato::LeastSquaresFunction<PhysicsT>>(aMesh, m_dataMap);
        for (Plato::OrdinalType tPropertyIndex = 0; tPropertyIndex < tPropertyNames.size(); ++tPropertyIndex)
        {
            const std::string   tPropertyName      = tPropertyNames[tPropertyIndex];
            const Plato::Scalar tPropertyWeight    = tPropertyWeights[tPropertyIndex];
            const Plato::Scalar tPropertyGoldValue = tPropertyGoldValues[tPropertyIndex];

            if (tPropertyName == "Mass")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(getMassFunction(aMesh, aMeshSets));
            }
            else if (tPropertyName == "CGx")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(
                      getFirstMomentOverMassRatio(aMesh, aMeshSets, "FirstX"));
            }
            else if (tPropertyName == "CGy")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(
                      getFirstMomentOverMassRatio(aMesh, aMeshSets, "FirstY"));
            }
            else if (tPropertyName == "CGz")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(
                      getFirstMomentOverMassRatio(aMesh, aMeshSets, "FirstZ"));
            }
            else if (tPropertyName == "Ixx")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(
                      getMomentOfInertia(aMesh, aMeshSets, "XX"));
            }
            else if (tPropertyName == "Iyy")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(
                      getMomentOfInertia(aMesh, aMeshSets, "YY"));
            }
            else if (tPropertyName == "Izz")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(
                      getMomentOfInertia(aMesh, aMeshSets, "ZZ"));
            }
            else if (tPropertyName == "Ixy")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(
                      getMomentOfInertia(aMesh, aMeshSets, "XY"));
            }
            else if (tPropertyName == "Ixz")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(
                      getMomentOfInertia(aMesh, aMeshSets, "XZ"));
            }
            else if (tPropertyName == "Iyz")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(
                      getMomentOfInertia(aMesh, aMeshSets, "YZ"));
            }
            else
            {
                const std::string tErrorString = std::string("Specified mass property '") +
                tPropertyName + "' not implemented. Options are: Mass, CGx, CGy, CGz, " 
                              + "Ixx, Iyy, Izz, Ixy, Ixz, Iyz";
                throw std::runtime_error(tErrorString);
            }
            mLeastSquaresFunction->appendFunctionWeight(tPropertyWeight);
            mLeastSquaresFunction->appendGoldFunctionValue(tPropertyGoldValue);
        }
    }

    /******************************************************************************//**
     * @brief Create the mass function only
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @return physics scalar function
    **********************************************************************************/
    std::shared_ptr<PhysicsScalarFunction<PhysicsT>>
    getMassFunction(Omega_h::Mesh& aMesh, 
                    Omega_h::MeshSets& aMeshSets)
    {
        std::shared_ptr<Plato::PhysicsScalarFunction<PhysicsT>> tMassFunction =
             std::make_shared<Plato::PhysicsScalarFunction<PhysicsT>>(aMesh, m_dataMap);
        tMassFunction->setFunctionName("Mass Function");

        std::string tCalculationType = std::string("Mass");

        std::shared_ptr<Plato::MassMoment<Residual>> tValue = 
             std::make_shared<Plato::MassMoment<Residual>>(aMesh, aMeshSets, m_dataMap);
        tValue->setMaterialDensity(mMaterialDensity);
        tValue->setCalculationType(tCalculationType);
        tMassFunction->allocateValue(tValue);

        std::shared_ptr<Plato::MassMoment<Jacobian>> tGradientU = 
             std::make_shared<Plato::MassMoment<Jacobian>>(aMesh, aMeshSets, m_dataMap);
        tGradientU->setMaterialDensity(mMaterialDensity);
        tGradientU->setCalculationType(tCalculationType);
        tMassFunction->allocateGradientU(tGradientU);

        std::shared_ptr<Plato::MassMoment<GradientZ>> tGradientZ = 
             std::make_shared<Plato::MassMoment<GradientZ>>(aMesh, aMeshSets, m_dataMap);
        tGradientZ->setMaterialDensity(mMaterialDensity);
        tGradientZ->setCalculationType(tCalculationType);
        tMassFunction->allocateGradientZ(tGradientZ);

        std::shared_ptr<Plato::MassMoment<GradientX>> tGradientX = 
             std::make_shared<Plato::MassMoment<GradientX>>(aMesh, aMeshSets, m_dataMap);
        tGradientX->setMaterialDensity(mMaterialDensity);
        tGradientX->setCalculationType(tCalculationType);
        tMassFunction->allocateGradientX(tGradientX);
        return tMassFunction;
    }

    /******************************************************************************//**
     * @brief Create the 'first mass moment divided by the mass' function (CG)
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aMomentType mass moment type (FirstX, FirstY, FirstZ)
     * @return scalar function base
    **********************************************************************************/
    std::shared_ptr<ScalarFunctionBase>
    getFirstMomentOverMassRatio(Omega_h::Mesh& aMesh, 
                           Omega_h::MeshSets& aMeshSets, 
                           const std::string & aMomentType)
    {
        const std::string tNumeratorName = std::string("CG Numerator (Moment type = ")
                                         + aMomentType + ")";
        std::shared_ptr<Plato::PhysicsScalarFunction<PhysicsT>> tNumerator =
             std::make_shared<Plato::PhysicsScalarFunction<PhysicsT>>(aMesh, m_dataMap);
        tNumerator->setFunctionName(tNumeratorName);

        std::shared_ptr<Plato::MassMoment<Residual>> tNumeratorValue = 
             std::make_shared<Plato::MassMoment<Residual>>(aMesh, aMeshSets, m_dataMap);
        tNumeratorValue->setMaterialDensity(mMaterialDensity);
        tNumeratorValue->setCalculationType(aMomentType);
        tNumerator->allocateValue(tNumeratorValue);

        std::shared_ptr<Plato::MassMoment<Jacobian>> tNumeratorGradientU = 
             std::make_shared<Plato::MassMoment<Jacobian>>(aMesh, aMeshSets, m_dataMap);
        tNumeratorGradientU->setMaterialDensity(mMaterialDensity);
        tNumeratorGradientU->setCalculationType(aMomentType);
        tNumerator->allocateGradientU(tNumeratorGradientU);

        std::shared_ptr<Plato::MassMoment<GradientZ>> tNumeratorGradientZ = 
             std::make_shared<Plato::MassMoment<GradientZ>>(aMesh, aMeshSets, m_dataMap);
        tNumeratorGradientZ->setMaterialDensity(mMaterialDensity);
        tNumeratorGradientZ->setCalculationType(aMomentType);
        tNumerator->allocateGradientZ(tNumeratorGradientZ);

        std::shared_ptr<Plato::MassMoment<GradientX>> tNumeratorGradientX = 
             std::make_shared<Plato::MassMoment<GradientX>>(aMesh, aMeshSets, m_dataMap);
        tNumeratorGradientX->setMaterialDensity(mMaterialDensity);
        tNumeratorGradientX->setCalculationType(aMomentType);
        tNumerator->allocateGradientX(tNumeratorGradientX);

        const std::string tDenominatorName = std::string("CG Mass Denominator (Moment type = ")
                                           + aMomentType + ")";
        std::shared_ptr<Plato::PhysicsScalarFunction<PhysicsT>> tDenominator = 
             getMassFunction(aMesh, aMeshSets);
        tDenominator->setFunctionName(tDenominatorName);

        std::shared_ptr<Plato::DivisionFunction<PhysicsT>> tMomentOverMassRatioFunction =
             std::make_shared<Plato::DivisionFunction<PhysicsT>>(aMesh, m_dataMap);
        tMomentOverMassRatioFunction->allocateNumeratorFunction(tNumerator);
        tMomentOverMassRatioFunction->allocateDenominatorFunction(tDenominator);
        tMomentOverMassRatioFunction->setFunctionName(std::string("CG ") + aMomentType);
        return tMomentOverMassRatioFunction;
    }

    /******************************************************************************//**
     * @brief Create the second mass moment function
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aMomentType second mass moment type (XX, XY, YY, ...)
     * @return scalar function base
    **********************************************************************************/
    std::shared_ptr<ScalarFunctionBase>
    getSecondMassMoment(Omega_h::Mesh& aMesh, 
                        Omega_h::MeshSets& aMeshSets, 
                        const std::string & aMomentType)
    {
        const std::string tInertiaName = std::string("Second Mass Moment (Moment type = ")
                                         + aMomentType + ")";
        std::shared_ptr<Plato::PhysicsScalarFunction<PhysicsT>> tSecondMomentFunction =
             std::make_shared<Plato::PhysicsScalarFunction<PhysicsT>>(aMesh, m_dataMap);
        tSecondMomentFunction->setFunctionName(tInertiaName);

        std::shared_ptr<Plato::MassMoment<Residual>> tValue = 
             std::make_shared<Plato::MassMoment<Residual>>(aMesh, aMeshSets, m_dataMap);
        tValue->setMaterialDensity(mMaterialDensity);
        tValue->setCalculationType(aMomentType);
        tSecondMomentFunction->allocateValue(tValue);

        std::shared_ptr<Plato::MassMoment<Jacobian>> tGradientU = 
             std::make_shared<Plato::MassMoment<Jacobian>>(aMesh, aMeshSets, m_dataMap);
        tGradientU->setMaterialDensity(mMaterialDensity);
        tGradientU->setCalculationType(aMomentType);
        tSecondMomentFunction->allocateGradientU(tGradientU);

        std::shared_ptr<Plato::MassMoment<GradientZ>> tGradientZ = 
             std::make_shared<Plato::MassMoment<GradientZ>>(aMesh, aMeshSets, m_dataMap);
        tGradientZ->setMaterialDensity(mMaterialDensity);
        tGradientZ->setCalculationType(aMomentType);
        tSecondMomentFunction->allocateGradientZ(tGradientZ);

        std::shared_ptr<Plato::MassMoment<GradientX>> tGradientX = 
             std::make_shared<Plato::MassMoment<GradientX>>(aMesh, aMeshSets, m_dataMap);
        tGradientX->setMaterialDensity(mMaterialDensity);
        tGradientX->setCalculationType(aMomentType);
        tSecondMomentFunction->allocateGradientX(tGradientX);

        return tSecondMomentFunction;
    }


    /******************************************************************************//**
     * @brief Create the moment of inertia function
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aAxes axes about which to compute the moment of inertia (XX, YY, ..)
     * @return scalar function base
    **********************************************************************************/
    std::shared_ptr<ScalarFunctionBase>
    getMomentOfInertia(Omega_h::Mesh& aMesh, 
                       Omega_h::MeshSets& aMeshSets, 
                       const std::string & aAxes)
    {
        std::shared_ptr<Plato::WeightedSumFunction<PhysicsT>> tMomentOfInertiaFunction = 
               std::make_shared<Plato::WeightedSumFunction<PhysicsT>>(aMesh, m_dataMap);
        tMomentOfInertiaFunction->setFunctionName(std::string("Inertia ") + aAxes);

        if (aAxes == "XX")
        {
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aMesh, aMeshSets, "SecondYY"));
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aMesh, aMeshSets, "SecondZZ"));
            tMomentOfInertiaFunction->appendFunctionWeight(1.0);
            tMomentOfInertiaFunction->appendFunctionWeight(1.0);
        }
        else if (aAxes == "YY")
        {
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aMesh, aMeshSets, "SecondXX"));
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aMesh, aMeshSets, "SecondZZ"));
            tMomentOfInertiaFunction->appendFunctionWeight(1.0);
            tMomentOfInertiaFunction->appendFunctionWeight(1.0);
        }
        else if (aAxes == "ZZ")
        {
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aMesh, aMeshSets, "SecondXX"));
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aMesh, aMeshSets, "SecondYY"));
            tMomentOfInertiaFunction->appendFunctionWeight(1.0);
            tMomentOfInertiaFunction->appendFunctionWeight(1.0);
        }
        else if (aAxes == "XY")
        {
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aMesh, aMeshSets, "SecondXY"));
            tMomentOfInertiaFunction->appendFunctionWeight(-1.0);
        }
        else if (aAxes == "XZ")
        {
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aMesh, aMeshSets, "SecondXZ"));
            tMomentOfInertiaFunction->appendFunctionWeight(-1.0);
        }
        else if (aAxes == "YZ")
        {
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aMesh, aMeshSets, "SecondYZ"));
            tMomentOfInertiaFunction->appendFunctionWeight(-1.0);
        }
        else
        {
            const std::string tErrorString = std::string("Specified axes '") +
            aAxes + "' not implemented for moment of inertia calculation. " 
                          + "Options are: XX, YY, ZZ, XY, XZ, YZ";
            throw std::runtime_error(tErrorString);
        }

        return tMomentOfInertiaFunction;
    }

public:
    /******************************************************************************//**
     * @brief Primary Mass Properties Function constructor
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     * @param [in] aInputParams input parameters database
     * @param [in] aName user defined function name
    **********************************************************************************/
    MassPropertiesFunction(Omega_h::Mesh& aMesh,
                           Omega_h::MeshSets& aMeshSets,
                           Plato::DataMap & aDataMap,
                           Teuchos::ParameterList& aInputParams,
                           std::string& aName) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            m_dataMap(aDataMap),
            mFunctionName(aName),
            mMaterialDensity(1.0)
    {
        initialize(aMesh, aMeshSets, aInputParams);
    }

    /******************************************************************************//**
     * @brief Update physics-based parameters within optimization iterations
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aState, const Plato::ScalarVector & aControl) const
    {
        mLeastSquaresFunction->updateProblem(aState, aControl);
    }

    /******************************************************************************//**
     * @brief Evaluate Mass Properties Function
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return scalar function evaluation
    **********************************************************************************/
    Plato::Scalar value(const Plato::ScalarVector & aState,
                        const Plato::ScalarVector & aControl,
                        Plato::Scalar aTimeStep = 0.0) const
    {
        Plato::Scalar tFunctionValue = mLeastSquaresFunction->value(aState, aControl, aTimeStep);
        return tFunctionValue;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the Mass Properties Function with respect to (wrt) the configuration parameters
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector gradient_x(const Plato::ScalarVector & aState,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    {
        Plato::ScalarVector tGradientX = mLeastSquaresFunction->gradient_x(aState, aControl, aTimeStep);
        return tGradientX;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the Mass Properties Function with respect to (wrt) the state variables
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector gradient_u(const Plato::ScalarVector & aState,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    {
        Plato::ScalarVector tGradientU = mLeastSquaresFunction->gradient_u(aState, aControl, aTimeStep);
        return tGradientU;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the Mass Properties Function with respect to (wrt) the control variables
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector gradient_z(const Plato::ScalarVector & aState,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    {
        Plato::ScalarVector tGradientZ = mLeastSquaresFunction->gradient_z(aState, aControl, aTimeStep);
        return tGradientZ;
    }

    /******************************************************************************//**
     * @brief Return user defined function name
     * @return User defined function name
    **********************************************************************************/
    std::string name() const
    {
        return mFunctionName;
    }
};
// class MassPropertiesFunction

}
//namespace Plato

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Electromechanics.hpp"
#include "Thermomechanics.hpp"

#ifdef PLATO_1D
extern template class Plato::MassPropertiesFunction<::Plato::Thermal<1>>;
extern template class Plato::MassPropertiesFunction<::Plato::Mechanics<1>>;
extern template class Plato::MassPropertiesFunction<::Plato::Electromechanics<1>>;
extern template class Plato::MassPropertiesFunction<::Plato::Thermomechanics<1>>;
#endif

#ifdef PLATO_2D
extern template class Plato::MassPropertiesFunction<::Plato::Thermal<2>>;
extern template class Plato::MassPropertiesFunction<::Plato::Mechanics<2>>;
extern template class Plato::MassPropertiesFunction<::Plato::Electromechanics<2>>;
extern template class Plato::MassPropertiesFunction<::Plato::Thermomechanics<2>>;
#endif

#ifdef PLATO_3D
extern template class Plato::MassPropertiesFunction<::Plato::Thermal<3>>;
extern template class Plato::MassPropertiesFunction<::Plato::Mechanics<3>>;
extern template class Plato::MassPropertiesFunction<::Plato::Electromechanics<3>>;
extern template class Plato::MassPropertiesFunction<::Plato::Thermomechanics<3>>;
#endif