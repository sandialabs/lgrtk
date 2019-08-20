#ifndef PLATO_PLASTICITY_HPP
#define PLATO_PLASTICITY_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/SimplexPlasticity.hpp"
#include "plato/J2PlasticityLocalResidual.hpp"
#include "plato/AnalyzeMacros.hpp"

namespace Plato
{

namespace PlasticityFactory
{

/******************************************************************************/
struct FunctionFactory
{
    /******************************************************************************//**
     * @brief Create a PLATO local vector function  inc (i.e. local residual equations)
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Analyze physics-based database
     * @param [in] aInputParams input parameters
     * @param [in] aFuncName vector function name
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<EvaluationType>>
    createLocalVectorFunctionInc(Omega_h::Mesh& aMesh, 
                                 Omega_h::MeshSets& aMeshSets,
                                 Plato::DataMap& aDataMap, 
                                 Teuchos::ParameterList& aInputParams,
                                 std::string aFuncName)
    {

        if(aFuncName == "J2Plasticity")
        {
          constexpr Plato::OrdinalType tSpaceDim = EvaluationType::SpatialDim;
          return std::make_shared
            <J2PlasticityLocalResidual<EvaluationType, Plato::SimplexPlasticity<tSpaceDim>>>
            (aMesh, aMeshSets, aDataMap, aInputParams);
        }
        else
        {
          const std::string tError = std::string("Unknown LocalVectorFunctionInc '") + aFuncName
                                   + "' specified.";
          THROWERR(tError)
        }
    }
}; // struct FunctionFactory

} // namespace PlasticityFactory


/****************************************************************************//**
 * @brief Concrete class for use as the PhysicsT template argument in VectorFunctionVMS
 *******************************************************************************/
template<Plato::OrdinalType SpaceDimParam>
class Plasticity: public Plato::SimplexPlasticity<SpaceDimParam>
{
public:
    typedef Plato::PlasticityFactory::FunctionFactory FunctionFactory;
    using SimplexT        = Plato::SimplexPlasticity<SpaceDimParam>;
    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
};

} // namespace Plato

#endif
