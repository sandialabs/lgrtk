#pragma once

#include <memory>
#include <cassert>
#include <vector>

#include <Omega_h_mesh.hpp>

#include "plato/WorksetBase.hpp"
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/ScalarFunctionBaseFactory.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

/******************************************************************************//**
 * @brief Division function class \f$ F(x) = numerator(x) / denominator(x) \f$
 **********************************************************************************/
template<typename PhysicsT>
class DivisionFunction : public Plato::ScalarFunctionBase, public Plato::WorksetBase<PhysicsT>
{
private:
    using Plato::WorksetBase<PhysicsT>::m_numDofsPerCell; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::m_numNodesPerCell; /*!< number of nodes per cell/element */
    using Plato::WorksetBase<PhysicsT>::m_numDofsPerNode; /*!< number of degree of freedom per node */
    using Plato::WorksetBase<PhysicsT>::m_numSpatialDims; /*!< number of spatial dimensions */
    using Plato::WorksetBase<PhysicsT>::m_numControl; /*!< number of control variables */
    using Plato::WorksetBase<PhysicsT>::m_numNodes; /*!< total number of nodes in the mesh */
    using Plato::WorksetBase<PhysicsT>::m_numCells; /*!< total number of cells/elements in the mesh */

    using Plato::WorksetBase<PhysicsT>::m_stateEntryOrdinal; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::m_controlEntryOrdinal; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::m_configEntryOrdinal; /*!< number of degree of freedom per cell/element */

    std::shared_ptr<Plato::ScalarFunctionBase> mScalarFunctionBaseNumerator; /*!< numerator function */
    std::shared_ptr<Plato::ScalarFunctionBase> mScalarFunctionBaseDenominator; /*!< denominator function */

    Plato::DataMap& m_dataMap; /*!< PLATO Engine and Analyze data map */

    std::string mFunctionName; /*!< User defined function name */

	/******************************************************************************//**
     * @brief Initialization of Division Function
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize (Omega_h::Mesh& aMesh, 
                     Omega_h::MeshSets& aMeshSets, 
                     Teuchos::ParameterList & aInputParams)
    {
        Plato::ScalarFunctionBaseFactory<PhysicsT> tFactory;

        auto tProblemFunctionName = aInputParams.sublist(mFunctionName);

        auto tNumeratorFunctionName = tProblemFunctionName.get<std::string>("Numerator");
        auto tDenominatorFunctionName = tProblemFunctionName.get<std::string>("Denominator");

        mScalarFunctionBaseNumerator = 
             tFactory.create(aMesh, aMeshSets, m_dataMap, aInputParams, tNumeratorFunctionName);

        mScalarFunctionBaseDenominator = 
             tFactory.create(aMesh, aMeshSets, m_dataMap, aInputParams, tDenominatorFunctionName);
    }

public:
    /******************************************************************************//**
     * @brief Primary division function constructor
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     * @param [in] aInputParams input parameters database
     * @param [in] aName user defined function name
    **********************************************************************************/
    DivisionFunction(Omega_h::Mesh& aMesh,
                Omega_h::MeshSets& aMeshSets,
                Plato::DataMap & aDataMap,
                Teuchos::ParameterList& aInputParams,
                std::string& aName) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            m_dataMap(aDataMap),
            mFunctionName(aName)
    {
        initialize(aMesh, aMeshSets, aInputParams);
    }

    /******************************************************************************//**
     * @brief Secondary division function constructor, used for unit testing
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
    **********************************************************************************/
    DivisionFunction(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            m_dataMap(aDataMap),
            mFunctionName("Division Function")
    {
    }

    /******************************************************************************//**
     * @brief Allocate numerator function base using the residual automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateNumeratorFunction(const std::shared_ptr<Plato::ScalarFunctionBase>& aInput)
    {
        mScalarFunctionBaseNumerator = aInput;
    }

    /******************************************************************************//**
     * @brief Allocate denominator function base using the residual automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateDenominatorFunction(const std::shared_ptr<Plato::ScalarFunctionBase>& aInput)
    {
        mScalarFunctionBaseDenominator = aInput;
    }

    /******************************************************************************//**
     * @brief Update physics-based parameters within optimization iterations
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aState, const Plato::ScalarVector & aControl) const
    {
        mScalarFunctionBaseNumerator->updateProblem(aState, aControl);
        mScalarFunctionBaseDenominator->updateProblem(aState, aControl);
    }

    /******************************************************************************//**
     * @brief Evaluate division function
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return scalar function evaluation
    **********************************************************************************/
    Plato::Scalar value(const Plato::ScalarVector & aState,
                        const Plato::ScalarVector & aControl,
                        Plato::Scalar aTimeStep = 0.0) const
    {
        Plato::Scalar tNumeratorValue = mScalarFunctionBaseNumerator->value(aState, aControl, aTimeStep);
        Plato::Scalar tDenominatorValue = mScalarFunctionBaseDenominator->value(aState, aControl, aTimeStep);
        Plato::Scalar tResult = tNumeratorValue / tDenominatorValue;
        if (tDenominatorValue == 0.0)
        {
            THROWERR("Denominator of division function evaluated to 0!")
        }
        
        return tResult;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the division function with respect to (wrt) the configuration parameters
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector gradient_x(const Plato::ScalarVector & aState,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    {
        const Plato::OrdinalType tNumDofs = m_numSpatialDims * m_numNodes;
        Plato::ScalarVector tGradientX ("gradient configuration", tNumDofs);

        Plato::Scalar tNumeratorValue = mScalarFunctionBaseNumerator->value(aState, aControl, aTimeStep);
        Plato::Scalar tDenominatorValue = mScalarFunctionBaseDenominator->value(aState, aControl, aTimeStep);

        Plato::ScalarVector tNumeratorGradX = mScalarFunctionBaseNumerator->gradient_x(aState, aControl, aTimeStep);
        Plato::ScalarVector tDenominatorGradX = mScalarFunctionBaseDenominator->gradient_x(aState, aControl, aTimeStep);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
        {
            tGradientX(tDof) = (tNumeratorGradX(tDof) * tDenominatorValue - 
                                tDenominatorGradX(tDof) * tNumeratorValue) 
                               / (pow(tDenominatorValue, 2));
        },"Division Function Grad X");
        return tGradientX;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the division function with respect to (wrt) the state variables
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector gradient_u(const Plato::ScalarVector & aState,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    {
        const Plato::OrdinalType tNumDofs = m_numDofsPerNode * m_numNodes;
        Plato::ScalarVector tGradientU ("gradient state", tNumDofs);

        Plato::Scalar tNumeratorValue = mScalarFunctionBaseNumerator->value(aState, aControl, aTimeStep);
        Plato::Scalar tDenominatorValue = mScalarFunctionBaseDenominator->value(aState, aControl, aTimeStep);

        Plato::ScalarVector tNumeratorGradU = mScalarFunctionBaseNumerator->gradient_u(aState, aControl, aTimeStep);
        Plato::ScalarVector tDenominatorGradU = mScalarFunctionBaseDenominator->gradient_u(aState, aControl, aTimeStep);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
        {
            tGradientU(tDof) = (tNumeratorGradU(tDof) * tDenominatorValue - 
                                tDenominatorGradU(tDof) * tNumeratorValue) 
                               / (pow(tDenominatorValue, 2));
        },"Division Function Grad U");
        return tGradientU;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the division function with respect to (wrt) the control variables
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector gradient_z(const Plato::ScalarVector & aState,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    {
        const Plato::OrdinalType tNumDofs = m_numNodes;
        Plato::ScalarVector tGradientZ ("gradient control", tNumDofs);
        
        Plato::Scalar tNumeratorValue = mScalarFunctionBaseNumerator->value(aState, aControl, aTimeStep);
        Plato::Scalar tDenominatorValue = mScalarFunctionBaseDenominator->value(aState, aControl, aTimeStep);

        Plato::ScalarVector tNumeratorGradZ = mScalarFunctionBaseNumerator->gradient_z(aState, aControl, aTimeStep);
        Plato::ScalarVector tDenominatorGradZ = mScalarFunctionBaseDenominator->gradient_z(aState, aControl, aTimeStep);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
        {
            tGradientZ(tDof) = (tNumeratorGradZ(tDof) * tDenominatorValue - 
                                tDenominatorGradZ(tDof) * tNumeratorValue) 
                               / (pow(tDenominatorValue, 2));
        },"Division Function Grad Z");

        return tGradientZ;
    }

    /******************************************************************************//**
     * @brief Set user defined function name
     * @param [in] function name
    **********************************************************************************/
    void setFunctionName(const std::string aFunctionName)
    {
        mFunctionName = aFunctionName;
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
// class DivisionFunction

}
//namespace Plato

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Electromechanics.hpp"
#include "Thermomechanics.hpp"

#ifdef PLATO_1D
extern template class Plato::DivisionFunction<::Plato::Thermal<1>>;
extern template class Plato::DivisionFunction<::Plato::Mechanics<1>>;
extern template class Plato::DivisionFunction<::Plato::Electromechanics<1>>;
extern template class Plato::DivisionFunction<::Plato::Thermomechanics<1>>;
#endif

#ifdef PLATO_2D
extern template class Plato::DivisionFunction<::Plato::Thermal<2>>;
extern template class Plato::DivisionFunction<::Plato::Mechanics<2>>;
extern template class Plato::DivisionFunction<::Plato::Electromechanics<2>>;
extern template class Plato::DivisionFunction<::Plato::Thermomechanics<2>>;
#endif

#ifdef PLATO_3D
extern template class Plato::DivisionFunction<::Plato::Thermal<3>>;
extern template class Plato::DivisionFunction<::Plato::Mechanics<3>>;
extern template class Plato::DivisionFunction<::Plato::Electromechanics<3>>;
extern template class Plato::DivisionFunction<::Plato::Thermomechanics<3>>;
#endif