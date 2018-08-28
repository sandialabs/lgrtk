#ifndef LGR_PLATO_CELL_FORCING_HPP
#define LGR_PLATO_CELL_FORCING_HPP

/******************************************************************************/
/*! Add forcing for homogenization cell problem.
  
    given a view, subtract the forcing column.
*/
/******************************************************************************/
namespace Plato {

template<int NumTerms>
class CellForcing
{
  private:

    const Omega_h::Matrix<NumTerms,NumTerms> m_cellStiffness;
    const int m_columnIndex;

  public:

    CellForcing( const Omega_h::Matrix<NumTerms,NumTerms> aCellStiffness, int aColumnIndex ) :
            m_cellStiffness(aCellStiffness), m_columnIndex(aColumnIndex) {}

    template<typename TensorScalarType>
    void
    add( Kokkos::View<TensorScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& tensor ) const
    {
      int numCells = tensor.extent(0);
      int numTerms = tensor.extent(1);
      auto cellStiffness = m_cellStiffness;
      auto& tTensor = tensor;
      auto columnIndex = m_columnIndex;
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numCells), LAMBDA_EXPRESSION(int cellOrdinal)
      {

        // add forcing
        //
        for(int iTerm=0; iTerm<numTerms; iTerm++){
          tTensor(cellOrdinal,iTerm) -= cellStiffness(iTerm, columnIndex);
        }
      }, "Add Forcing");
    }
};

} // end namespace Plato
#endif
