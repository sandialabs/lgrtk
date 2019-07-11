#ifndef PLATO_MATH_FUNCTORS
#define PLATO_MATH_FUNCTORS

namespace Plato {
 
    /******************************************************************************//**
     * @brief Functor for computing the row sum of a given matrix
    **********************************************************************************/
    template<int NumDofsPerNode_I, int NumDofsPerNode_J=NumDofsPerNode_I>
    class RowSum {
        private:
            const typename Plato::CrsMatrixType::RowMapVector mRowMap;
            const typename Plato::CrsMatrixType::ScalarVector mEntries;

        public:
            /**********************************************************************//**
            * @brief Constructor
            * @param [in] aMatrix Matrix for witch the row sum will be computed
            **************************************************************************/
            RowSum(Teuchos::RCP<Plato::CrsMatrixType> aMatrix) : 
                mRowMap(aMatrix->rowMap()),
                mEntries(aMatrix->entries()) {}

            /**********************************************************************//**
            * @brief Functor
            * @param [in]  blockRowOrdinal Ordinal for the row for which the sum is to be computed
            * @param [out] aRowSum Row sum vector (assumed initialized to zero)
            **************************************************************************/
            DEVICE_TYPE inline void
            operator()( Plato::OrdinalType blockRowOrdinal, 
                        Plato::ScalarVector aRowSum ) const
            {
                using Plato::OrdinalType;
                using Plato::RowMapEntryType;

                // for each entry in this block row
                RowMapEntryType tRowStart = mRowMap(blockRowOrdinal  );
                RowMapEntryType tRowEnd   = mRowMap(blockRowOrdinal+1);
 
                OrdinalType tTotalBlockSize = NumDofsPerNode_I*NumDofsPerNode_J;

                for (OrdinalType tColNodeOrd=tRowStart; tColNodeOrd<tRowEnd; tColNodeOrd++)
                {

                    OrdinalType tEntryOrdinalOffset = tTotalBlockSize * tColNodeOrd;

                    // for each row in this block
                    for(OrdinalType idim=0; idim<NumDofsPerNode_I; idim++)
                    {
                        // for each col in this block
                        OrdinalType tVectorOrdinal = blockRowOrdinal*NumDofsPerNode_I + idim;
                        OrdinalType tMatrixOrdinal = tEntryOrdinalOffset + NumDofsPerNode_J*idim;
                        for(OrdinalType jdim=0; jdim<NumDofsPerNode_J; jdim++)
                        {
                            aRowSum(tVectorOrdinal) += mEntries(tMatrixOrdinal + jdim);
                        }
                    }
                }
            }
    };

    /******************************************************************************//**
     * @brief Functor for computing the weighted inverse
    **********************************************************************************/
    template<int NumDofsPerNode_I, int NumDofsPerNode_J=NumDofsPerNode_I>
    class InverseWeight {

        public:
            /**********************************************************************//**
            * @brief Functor
            * @param [in]  blockRowOrdinal Ordinal for the row for which the sum is to be computed
            * @param [in]  Row sum vector, R
            * @param [in]  Input vector, b
            * @param [out] Output vector, x
            *
            *  x[i] = b[i] / R[i]
            *
            **************************************************************************/
            DEVICE_TYPE inline void
            operator()( Plato::OrdinalType blockRowOrdinal, 
                        Plato::ScalarVector aRowSum,
                        Plato::ScalarVector aRHS,
                        Plato::ScalarVector aLHS, Plato::Scalar scale=1.0 ) const
            {
                // for each row in this block
                for(Plato::OrdinalType idim=0; idim<NumDofsPerNode_I; idim++)
                {
                    // for each col in this block
                    Plato::OrdinalType tVectorOrdinal = blockRowOrdinal*NumDofsPerNode_I + idim;
                    aLHS(tVectorOrdinal) = (aRHS(tVectorOrdinal) / aRowSum(tVectorOrdinal)) * scale;
                }
            }
    };

} // namespace Plato

#endif
