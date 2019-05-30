#pragma once

#include "plato/PlatoStaticsTypes.hpp"
#include "plato/SimplexMechanics.hpp"

namespace Plato
{

/******************************************************************************/
/*! Tensile energy density functor.
 *  
 *  Given principal strains and lame constants, return the tensile energy density
 *  (Assumes isotropic linear elasticity. In 2D assumes plane strain!)
 */
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class TensileEnergyDensity : public Plato::SimplexMechanics<SpaceDim>
{
  public:

    template<typename StrainType, typename ResultType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType aCellOrdinal,
                const Plato::ScalarMultiVectorT<StrainType> & aPrincipalStrains,
                const Plato::Scalar & aLameLambda,
                const Plato::Scalar & aLameMu,
                const Plato::ScalarVectorT<ResultType> & aTensileEnergyDensity) const 
    {
        ResultType tTensileEnergyDensity = static_cast<Plato::Scalar>(0.0);
        StrainType tStrainTrace = static_cast<Plato::Scalar>(0.0);
        for (Plato::OrdinalType tDim = 0; tDim < SpaceDim; ++tDim)
        {
            tStrainTrace += aPrincipalStrains(aCellOrdinal, tDim);
            if (aPrincipalStrains(aCellOrdinal, tDim) >= 0.0)
            {
                tTensileEnergyDensity += (aPrincipalStrains(aCellOrdinal, tDim) * 
                                          aPrincipalStrains(aCellOrdinal, tDim) * aLameMu);
            }
        }
        StrainType tStrainTraceTensile = (tStrainTrace >= 0.0) ? tStrainTrace : static_cast<Plato::Scalar>(0.0);
        tTensileEnergyDensity += (aLameLambda * tStrainTraceTensile * 
                                                tStrainTraceTensile * static_cast<Plato::Scalar>(0.5));
        aTensileEnergyDensity(aCellOrdinal) = tTensileEnergyDensity;
    }
};
//class TensileEnergyDensity


}
//namespace Plato

#ifdef PLATO_1D
extern template class Plato::TensileEnergyDensity<1>;
#endif

#ifdef PLATO_2D
extern template class Plato::TensileEnergyDensity<2>;
#endif

#ifdef PLATO_3D
extern template class Plato::TensileEnergyDensity<3>;
#endif