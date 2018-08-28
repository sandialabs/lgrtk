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

class AbstractProblem
{
public:
    virtual ~AbstractProblem()
    {
    }

    virtual Plato::ScalarVector getAdjoint()=0;
    virtual Plato::ScalarMultiVector getState()=0;
    virtual void setState(const Plato::ScalarMultiVector & aState)=0;

    // Functions associated with residual
    virtual void
    applyConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aVector)=0;

    virtual void
    applyBoundaryLoads(const Plato::ScalarVector & aForce)=0;

    virtual Plato::ScalarMultiVector
    solution(const Plato::ScalarVector & aControl)=0;

    // Functions associated with constraint
    virtual Plato::Scalar
    constraintValue(const Plato::ScalarVector & aControl)=0;

    virtual Plato::Scalar
    constraintValue(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)=0;

    virtual Plato::ScalarVector
    constraintGradient(const Plato::ScalarVector & aControl)=0;

    virtual Plato::ScalarVector
    constraintGradient(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)=0;

    virtual Plato::ScalarVector
    constraintGradientX(const Plato::ScalarVector & aControl)=0;

    virtual Plato::ScalarVector
    constraintGradientX(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)=0;

    // Functions associated with objective
    virtual Plato::Scalar
    objectiveValue(const Plato::ScalarVector & aControl)=0;

    virtual Plato::Scalar
    objectiveValue(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)=0;

    virtual Plato::ScalarVector
    objectiveGradient(const Plato::ScalarVector & aControl)=0;

    virtual Plato::ScalarVector
    objectiveGradient(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)=0;

    virtual Plato::ScalarVector
    objectiveGradientX(const Plato::ScalarVector & aControl)=0;

    virtual Plato::ScalarVector
    objectiveGradientX(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)=0;

    Plato::DataMap mDataMap;
    decltype(mDataMap)& getDataMap()
    {
        return mDataMap;
    }
};
// end class AbstractProblem

}// end namespace Plato

#endif /* PLATOABSTRACTPROBLEM_HPP_ */
