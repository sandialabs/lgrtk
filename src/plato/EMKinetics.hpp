#ifndef LGR_PLATO_EMKINETICS_HPP
#define LGR_PLATO_EMKINETICS_HPP

#include "plato/SimplexElectromechanics.hpp"
#include "plato/LinearElectroelasticMaterial.hpp"

/******************************************************************************/
/*! Electroelastics functor.
  
    given a strain and electric field, compute the stress and electric displacement

    IMPORTANT NOTE:  This model is scaled to make the coupling better conditioned. 
    The second equation is multiplied by a:

     i.e., this:     | T |   |   C     -e  |  |  S  |
                     |   | = |             |  |     |
                     | D |   |   e      p  |  |  E  |
              
     becomes this:   | T |   |   C   -a*e  |  |  S  |
                     |   | = |             |  |     |
                     |a*D|   |  a*e  a*a*p |  | E/a |
              
     A typical value for a is 1e9.  So, this model computes (T  a*D) from 
     (S E/a) which means that electrical quantities in the simulation are scaled:
 
            Electric potential:      phi/a
            Electric field:          E/a
            Electric displacement:   D*q
            Electric charge density: a*q

     and should be 'unscaled' before writing output.  Further, boundary conditions
     must be scaled.

    IMPORTANT NOTE 2:  This model is not positive definite!

*/
/******************************************************************************/

namespace Plato
{

template<int SpaceDim>
class EMKinetics : public Plato::SimplexElectromechanics<SpaceDim>
{
  private:

    using Plato::SimplexElectromechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::SimplexElectromechanics<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexElectromechanics<SpaceDim>::mNumDofsPerNode;
    using Plato::SimplexElectromechanics<SpaceDim>::mNumDofsPerCell;

    const Omega_h::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellStiffness;
    const Omega_h::Matrix<SpaceDim, mNumVoigtTerms> mCellPiezoelectricCoupling;
    const Omega_h::Matrix<SpaceDim, SpaceDim> mCellPermittivity;
 
    const Plato::Scalar mAlpha;
    const Plato::Scalar mAlpha2;

  public:

    EMKinetics( const Teuchos::RCP<Plato::LinearElectroelasticMaterial<SpaceDim>> materialModel ) :
            mCellStiffness(materialModel->getStiffnessMatrix()),
            mCellPiezoelectricCoupling(materialModel->getPiezoMatrix()),
            mCellPermittivity(materialModel->getPermittivityMatrix()),
            mAlpha(materialModel->getAlpha()),
            mAlpha2(mAlpha*mAlpha) { }

    template<typename KineticsScalarType, typename KinematicsScalarType>
    DEVICE_TYPE inline void
    operator()( int cellOrdinal,
                Kokkos::View<KineticsScalarType**,   Kokkos::LayoutRight, Plato::MemSpace> const& stress,
                Kokkos::View<KineticsScalarType**,   Kokkos::LayoutRight, Plato::MemSpace> const& edisp,
                Kokkos::View<KinematicsScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& strain,
                Kokkos::View<KinematicsScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& efield) const {

      // compute stress
      //
      for( int iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
        stress(cellOrdinal,iVoigt) = 0.0;
        for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
          stress(cellOrdinal,iVoigt) += strain(cellOrdinal,jVoigt)*mCellStiffness(iVoigt, jVoigt);
        }
        for( int jDim=0; jDim<SpaceDim; jDim++){
          stress(cellOrdinal,iVoigt) -= mAlpha*efield(cellOrdinal,jDim)*mCellPiezoelectricCoupling(jDim, iVoigt);
        }
      }

      // compute edisp
      //
      for( int iDim=0; iDim<SpaceDim; iDim++){
        edisp(cellOrdinal,iDim) = 0.0;
        for( int jDim=0; jDim<SpaceDim; jDim++){
          edisp(cellOrdinal,iDim) += mAlpha2*efield(cellOrdinal,jDim)*mCellPermittivity(iDim, jDim);
        }
        for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
          edisp(cellOrdinal,iDim) += mAlpha*strain(cellOrdinal,jVoigt)*mCellPiezoelectricCoupling(iDim, jVoigt);
        }
      }
    }
};

} // namespace Plato

#endif
