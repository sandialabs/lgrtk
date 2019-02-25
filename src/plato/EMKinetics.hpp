#ifndef LGR_PLATO_KINETICS_HPP
#define LGR_PLATO_KINETICS_HPP

#include "plato/SimplexElectromechanics.hpp"
#include "plato/LinearElectroelasticMaterial.hpp"

/******************************************************************************/
/*! Electroelastics functor.
  
    given a strain and electric field, compute the stress and electric displacement

    IMPORTANT NOTE:  This model is scaled to make the coupling symmetric and better
     conditioned.  The second equation is multiplied by -a:

     i.e., this:     |  T |   |   C     -e  |  |  S  |
                     |    | = |             |  |     |
                     |  D |   |   e      p  |  |  E  |
              
     becomes this:   |  T |   |   C   -a*e  |  |  S  |
                     |    | = |             |  |     |
                     |-a*D|   | -a*e -a*a*p |  | E/a |
              
     A typical value for a is 1e9.  So, this model computes (T -a*D) from 
     (S E/a) which means that electrical quantities in the simulation are scaled:
 
            Electric potential:      phi/a
            Electric field:          E/a
            Electric displacement:   -D*q
            Electric charge density: -a*q

     and should be 'unscaled' before writing output.  Further, boundary conditions
     must be scaled.

    IMPORTANT NOTE 2:  This model is not positive definite!

*/
/******************************************************************************/
template<int SpaceDim>
class EMKinetics : public Plato::SimplexElectromechanics<SpaceDim>
{
  private:

    using Plato::SimplexElectromechanics<SpaceDim>::m_numVoigtTerms;
    using Plato::SimplexElectromechanics<SpaceDim>::m_numNodesPerCell;
    using Plato::SimplexElectromechanics<SpaceDim>::m_numDofsPerNode;
    using Plato::SimplexElectromechanics<SpaceDim>::m_numDofsPerCell;

    const Omega_h::Matrix<m_numVoigtTerms,m_numVoigtTerms> m_cellStiffness;
    const Omega_h::Matrix<SpaceDim, m_numVoigtTerms> m_cellPiezoelectricCoupling;
    const Omega_h::Matrix<SpaceDim, SpaceDim> m_cellPermittivity;
 
    const Plato::Scalar m_alpha;
    const Plato::Scalar m_alpha2;

  public:

    EMKinetics( const Teuchos::RCP<Plato::LinearElectroelasticMaterial<SpaceDim>> materialModel ) :
            m_cellStiffness(materialModel->getStiffnessMatrix()),
            m_cellPiezoelectricCoupling(materialModel->getPiezoMatrix()),
            m_cellPermittivity(materialModel->getPermittivityMatrix()),
            m_alpha(materialModel->getAlpha()),
            m_alpha2(m_alpha*m_alpha) { }

    template<typename KineticsScalarType, typename KinematicsScalarType>
    DEVICE_TYPE inline void
    operator()( int cellOrdinal,
                Kokkos::View<KineticsScalarType**,   Kokkos::LayoutRight, Plato::MemSpace> const& stress,
                Kokkos::View<KineticsScalarType**,   Kokkos::LayoutRight, Plato::MemSpace> const& edisp,
                Kokkos::View<KinematicsScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& strain,
                Kokkos::View<KinematicsScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& efield) const {

      // compute stress
      //
      for( int iVoigt=0; iVoigt<m_numVoigtTerms; iVoigt++){
        stress(cellOrdinal,iVoigt) = 0.0;
        for( int jVoigt=0; jVoigt<m_numVoigtTerms; jVoigt++){
          stress(cellOrdinal,iVoigt) += strain(cellOrdinal,jVoigt)*m_cellStiffness(iVoigt, jVoigt);
        }
        for( int jDim=0; jDim<SpaceDim; jDim++){
          stress(cellOrdinal,iVoigt) -= m_alpha*efield(cellOrdinal,jDim)*m_cellPiezoelectricCoupling(jDim, iVoigt);
        }
      }

      // compute edisp
      //
      for( int iDim=0; iDim<SpaceDim; iDim++){
        edisp(cellOrdinal,iDim) = 0.0;
        for( int jDim=0; jDim<SpaceDim; jDim++){
          edisp(cellOrdinal,iDim) += (-m_alpha2)*efield(cellOrdinal,jDim)*m_cellPermittivity(iDim, jDim);
        }
        for( int jVoigt=0; jVoigt<m_numVoigtTerms; jVoigt++){
          edisp(cellOrdinal,iDim) += (-m_alpha)*strain(cellOrdinal,jVoigt)*m_cellPiezoelectricCoupling(iDim, jVoigt);
        }
      }
    }
};
#endif
