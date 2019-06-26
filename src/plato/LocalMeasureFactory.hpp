#pragma once

#include <memory>

#include "plato/SimplexFadTypes.hpp"
#include "AnalyzeMacros.hpp"

#include "plato/VonMisesLocalMeasure.hpp"
#include "plato/TensileEnergyDensityLocalMeasure.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

/**********************************************************************************/
template<typename EvaluationType>
class LocalMeasureFactory
{
/**********************************************************************************/
public:
    LocalMeasureFactory (){}
    ~LocalMeasureFactory (){}
    std::shared_ptr<Plato::AbstractLocalMeasure<EvaluationType>> 
    create(Teuchos::ParameterList& aInputParams, const std::string & aFuncName)
    {
        auto tFunctionSpecs = aInputParams.sublist(aFuncName);
        auto tLocalMeasure = tFunctionSpecs.get<std::string>("Local Measure", "VonMises");

        if(tLocalMeasure == "VonMises")
        {
            return std::make_shared<VonMisesLocalMeasure<EvaluationType>>(aInputParams, "VonMises");
        }
        else if(tLocalMeasure == "TensileEnergyDensity")
        {
            return std::make_shared<TensileEnergyDensityLocalMeasure<EvaluationType>>
                                                             (aInputParams, "TensileEnergyDensity");
        }
        else
        {
            THROWERR("Unknown 'Local Measure' specified in 'Plato Problem' ParameterList")
        }
    }
};
// class LocalMeasureFactory

}
//namespace Plato

#include "plato/SimplexMechanics.hpp"

#ifdef PLATO_1D
extern template class Plato::LocalMeasureFactory<Plato::ResidualTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::LocalMeasureFactory<Plato::JacobianTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::LocalMeasureFactory<Plato::GradientXTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::LocalMeasureFactory<Plato::GradientZTypes<Plato::SimplexMechanics<1>>>;
#endif

#ifdef PLATO_2D
extern template class Plato::LocalMeasureFactory<Plato::ResidualTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::LocalMeasureFactory<Plato::JacobianTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::LocalMeasureFactory<Plato::GradientXTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::LocalMeasureFactory<Plato::GradientZTypes<Plato::SimplexMechanics<2>>>;
#endif

#ifdef PLATO_3D
extern template class Plato::LocalMeasureFactory<Plato::ResidualTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::LocalMeasureFactory<Plato::JacobianTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::LocalMeasureFactory<Plato::GradientXTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::LocalMeasureFactory<Plato::GradientZTypes<Plato::SimplexMechanics<3>>>;
#endif