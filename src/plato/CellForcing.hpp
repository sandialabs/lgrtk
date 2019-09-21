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

    const Omega_h::Matrix<NumTerms,NumTerms> mCellStiffness;
    const int mColumnIndex;

  public:

    CellForcing( const Omega_h::Matrix<NumTerms,NumTerms> aCellStiffness, int aColumnIndex ) :
            mCellStiffness(aCellStiffness), mColumnIndex(aColumnIndex) {}

    template<typename TensorScalarType>
    void
    add( Kokkos::View<TensorScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& tensor ) const
    {
      int numCells = tensor.extent(0);
      int numTerms = tensor.extent(1);
      auto cellStiffness = mCellStiffness;
      auto& tTensor = tensor;
      auto columnIndex = mColumnIndex;
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
