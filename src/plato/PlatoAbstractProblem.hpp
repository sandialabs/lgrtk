/*
 * PlatoAbstractProblem.hpp
 *
 *  Created on: April 19, 2018
 */

#ifndef PLATOABSTRACTPROBLEM_HPP_
#define PLATOABSTRACTPROBLEM_HPP_

#include <Teuchos_RCPDecl.hpp>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

struct partial
{
    enum derivative_t
    {
        CONTROL = 0, STATE = 1, CONFIGURATION = 2,
    };
}; // end struct partial

/******************************************************************************//**
 * @brief Abstract interface for a PLATO problem
**********************************************************************************/
class AbstractProblem
{
public:
    /******************************************************************************//**
     * @brief PLATO abstract problem destructor
    **********************************************************************************/
    virtual ~AbstractProblem()
    {
    }

    /******************************************************************************//**
     * @brief Return 2D view of adjoint variables
     * @return 2D view of adjoint variables
    **********************************************************************************/
    virtual Plato::ScalarMultiVector getAdjoint()=0;

    /******************************************************************************//**
     * @brief Return 2D view of state variables
     * @return aState 2D view of state variables
    **********************************************************************************/
    virtual Plato::ScalarMultiVector getState()=0;

    /******************************************************************************//**
     * @brief Set state variables
     * @param [in] aState 2D view of state variables
    **********************************************************************************/
    virtual void setState(const Plato::ScalarMultiVector & aState)=0;

    /******************************************************************************//**
     * @brief Apply Dirichlet constraints
     * @param [in] aMatrix Compressed Row Storage (CRS) matrix
     * @param [in] aVector 1D view of Right-Hand-Side forces
    **********************************************************************************/
    virtual void
    applyConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aVector)=0;

    /******************************************************************************//**
     * @brief Apply boundary forces
     * @param [in/out] aForce 1D view of forces
    **********************************************************************************/
    virtual void
    applyBoundaryLoads(const Plato::ScalarVector & aForce)=0;

    /******************************************************************************//**
     * @brief Update physics-based parameters within optimization iterations
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 1D container of control variables
    **********************************************************************************/
    virtual void updateProblem(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)=0;

    /******************************************************************************//**
     * @brief Solve system of equations
     * @param [in] aControl 1D view of control variables
     * @return 2D view of state variables
    **********************************************************************************/
    virtual Plato::ScalarMultiVector
    solution(const Plato::ScalarVector & aControl)=0;

    /******************************************************************************//**
     * @brief Evaluate constraint function
     * @param [in] aControl 1D view of control variables
     * @return constraint function value
    **********************************************************************************/
    virtual Plato::Scalar
    constraintValue(const Plato::ScalarVector & aControl)=0;

    /******************************************************************************//**
     * @brief Evaluate constraint function
     * @param [in] aControl 1D view of control variables
     * @param [in] aState 2D view of state variables
     * @return constraint function value
    **********************************************************************************/
    virtual Plato::Scalar
    constraintValue(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)=0;

    /******************************************************************************//**
     * @brief Evaluate constraint partial derivative wrt control variables
     * @param [in] aControl 1D view of control variables
     * @return 1D view - constraint partial derivative wrt control variables
    **********************************************************************************/
    virtual Plato::ScalarVector
    constraintGradient(const Plato::ScalarVector & aControl)=0;

    /******************************************************************************//**
     * @brief Evaluate constraint partial derivative wrt control variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aState 2D view of state variables
     * @return 1D view - constraint partial derivative wrt control variables
    **********************************************************************************/
    virtual Plato::ScalarVector
    constraintGradient(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)=0;

    /******************************************************************************//**
     * @brief Evaluate constraint partial derivative wrt configuration variables
     * @param [in] aControl 1D view of control variables
     * @return 1D view - constraint partial derivative wrt configuration variables
    **********************************************************************************/
    virtual Plato::ScalarVector
    constraintGradientX(const Plato::ScalarVector & aControl)=0;

    /******************************************************************************//**
     * @brief Evaluate constraint partial derivative wrt configuration variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aState 2D view of state variables
     * @return 1D view - constraint partial derivative wrt configuration variables
    **********************************************************************************/
    virtual Plato::ScalarVector
    constraintGradientX(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)=0;

    /******************************************************************************//**
     * @brief Evaluate objective function
     * @param [in] aControl 1D view of control variables
     * @return objective function value
    **********************************************************************************/
    virtual Plato::Scalar
    objectiveValue(const Plato::ScalarVector & aControl)=0;

    /******************************************************************************//**
     * @brief Evaluate objective function
     * @param [in] aControl 1D view of control variables
     * @param [in] aState 2D view of state variables
     * @return objective function value
    **********************************************************************************/
    virtual Plato::Scalar
    objectiveValue(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)=0;

    /******************************************************************************//**
     * @brief Evaluate objective partial derivative wrt control variables
     * @param [in] aControl 1D view of control variables
     * @return 1D view - objective partial derivative wrt control variables
    **********************************************************************************/
    virtual Plato::ScalarVector
    objectiveGradient(const Plato::ScalarVector & aControl)=0;

    /******************************************************************************//**
     * @brief Evaluate objective gradient wrt control variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aState 2D view of state variables
     * @return 1D view - objective gradient wrt control variables
    **********************************************************************************/
    virtual Plato::ScalarVector
    objectiveGradient(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)=0;

    /******************************************************************************//**
     * @brief Evaluate objective partial derivative wrt configuration variables
     * @param [in] aControl 1D view of control variables
     * @return 1D view - objective partial derivative wrt configuration variables
    **********************************************************************************/
    virtual Plato::ScalarVector
    objectiveGradientX(const Plato::ScalarVector & aControl)=0;

    /******************************************************************************//**
     * @brief Evaluate objective gradient wrt configuration variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aState 2D view of state variables
     * @return 1D view - objective gradient wrt configuration variables
    **********************************************************************************/
    virtual Plato::ScalarVector
    objectiveGradientX(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)=0;

    /******************************************************************************//**
     * @brief Return PLATO Analyze data map that enables import/export rights to PLATO Engine
     * @return PLATO Analyze data map
    **********************************************************************************/
    Plato::DataMap mDataMap;
    decltype(mDataMap)& getDataMap()
    {
        return mDataMap;
    }
};
// end class AbstractProblem

}// end namespace Plato

#endif /* PLATOABSTRACTPROBLEM_HPP_ */
