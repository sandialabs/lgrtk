#ifndef PLATO_MATH_FUNCTORS
#define PLATO_MATH_FUNCTORS

namespace Plato {
 
    /******************************************************************************//**
     * @brief Functor for computing the row sum of a given matrix
    **********************************************************************************/
    class RowSum {
        private:
            const typename Plato::CrsMatrixType::RowMapVector mRowMap;
            const typename Plato::CrsMatrixType::ScalarVector mEntries;
            const Plato::OrdinalType mNumDofsPerNode_I;
            const Plato::OrdinalType mNumDofsPerNode_J;

        public:
            /**********************************************************************//**
            * @brief Constructor
            * @param [in] aMatrix Matrix for witch the row sum will be computed
            **************************************************************************/
            RowSum(Teuchos::RCP<Plato::CrsMatrixType> aMatrix) : 
                mRowMap(aMatrix->rowMap()),
                mEntries(aMatrix->entries()),
                mNumDofsPerNode_I(aMatrix->blockSizeRow()),
                mNumDofsPerNode_J(aMatrix->blockSizeCol()) {}

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

                // for each entry in this block row
                OrdinalType tRowStart = mRowMap(blockRowOrdinal  );
                OrdinalType tRowEnd   = mRowMap(blockRowOrdinal+1);
 
                OrdinalType tTotalBlockSize = mNumDofsPerNode_I*mNumDofsPerNode_J;

                for (OrdinalType tColNodeOrd=tRowStart; tColNodeOrd<tRowEnd; tColNodeOrd++)
                {

                    OrdinalType tEntryOrdinalOffset = tTotalBlockSize * tColNodeOrd;

                    // for each row in this block
                    for(OrdinalType idim=0; idim<mNumDofsPerNode_I; idim++)
                    {
                        // for each col in this block
                        OrdinalType tVectorOrdinal = blockRowOrdinal*mNumDofsPerNode_I + idim;
                        OrdinalType tMatrixOrdinal = tEntryOrdinalOffset + mNumDofsPerNode_J*idim;
                        for(OrdinalType jdim=0; jdim<mNumDofsPerNode_J; jdim++)
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

    /******************************************************************************//**
     * @brief Functor for computing the row sum of a given matrix
    **********************************************************************************/
    class DiagonalInverseMultiply {
        private:
            const typename Plato::CrsMatrixType::RowMapVector mRowMap;
            const typename Plato::CrsMatrixType::ScalarVector mEntries;
            const Plato::OrdinalType mNumDofsPerNode_I;
            const Plato::OrdinalType mNumDofsPerNode_J;

        public:
            /**********************************************************************//**
            * @brief Constructor
            * @param [in] aMatrix Matrix to witch the inverse diagonal multiply will be applied
            **************************************************************************/
            DiagonalInverseMultiply(Teuchos::RCP<Plato::CrsMatrixType> aMatrix) : 
                mRowMap(aMatrix->rowMap()),
                mEntries(aMatrix->entries()),
                mNumDofsPerNode_I(aMatrix->blockSizeRow()),
                mNumDofsPerNode_J(aMatrix->blockSizeCol()) {}

            /**********************************************************************//**
            * @brief Functor
            * @param [in]  blockRowOrdinal Ordinal for the block row to which the inverse diagonal multiply is applied
            * @param [out] aDiagonals Vector of diagonal entries
            **************************************************************************/
            DEVICE_TYPE inline void
            operator()( Plato::OrdinalType blockRowOrdinal, 
                        Plato::ScalarVector aDiagonals ) const
            {
                using Plato::OrdinalType;

                // for each entry in this block row
                OrdinalType tRowStart = mRowMap(blockRowOrdinal  );
                OrdinalType tRowEnd   = mRowMap(blockRowOrdinal+1);
 
                OrdinalType tTotalBlockSize = mNumDofsPerNode_I*mNumDofsPerNode_J;

                for (OrdinalType tColNodeOrd=tRowStart; tColNodeOrd<tRowEnd; tColNodeOrd++)
                {

                    OrdinalType tEntryOrdinalOffset = tTotalBlockSize * tColNodeOrd;

                    // for each row in this block
                    for(OrdinalType idim=0; idim<mNumDofsPerNode_I; idim++)
                    {
                        // for each col in this block
                        OrdinalType tVectorOrdinal = blockRowOrdinal*mNumDofsPerNode_I + idim;
                        OrdinalType tMatrixOrdinal = tEntryOrdinalOffset + mNumDofsPerNode_J*idim;
                        for(OrdinalType jdim=0; jdim<mNumDofsPerNode_J; jdim++)
                        {
                            mEntries(tMatrixOrdinal + jdim) /= aDiagonals(tVectorOrdinal);
                        }
                    }
                }
            }
    };

} // namespace Plato

#endif
