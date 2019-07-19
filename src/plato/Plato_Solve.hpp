#ifndef PLATO_SOLVE_HPP
#define PLATO_SOLVE_HPP

#include "plato/PlatoMathFunctors.hpp"

#ifdef HAVE_AMGX
#include "plato/alg/AmgXSparseLinearProblem.hpp"
#endif

namespace Plato {

namespace Solve {

    /******************************************************************************//**
     * @brief Solve linear system, A x = b.
     * @param [in]     a_A Matrix, A
     * @param [in/out] a_x Solution vector, x, with initial guess
     * @param [in]     a_b Forcing vector, b
    **********************************************************************************/
    template <Plato::OrdinalType NumDofsPerNode>
    void Consistent(
        Teuchos::RCP<Plato::CrsMatrixType> a_A, 
        Plato::ScalarVector a_x,
        Plato::ScalarVector a_b)
        {
#ifdef HAVE_AMGX
              using AmgXLinearProblem = Plato::AmgXSparseLinearProblem< Plato::OrdinalType, NumDofsPerNode>;
              auto tConfigString = AmgXLinearProblem::getConfigString();
              auto tSolver = Teuchos::rcp(new AmgXLinearProblem(*a_A, a_x, a_b, tConfigString));
              tSolver->solve();
              tSolver = Teuchos::null;
#endif
        }

    /******************************************************************************//**
     * @brief Approximate solution fo linear system, A x = b, by x = R^-1 b, where
     *        R is the row sum of A.
     * @param [in]     a_A Matrix, A
     * @param [in/out] a_x Solution vector, x, with initial guess
     * @param [in]     a_b Forcing vector, b
    **********************************************************************************/
    template <Plato::OrdinalType NumDofsPerNode>
    void RowSummed(
        Teuchos::RCP<Plato::CrsMatrixType> a_A, 
        Plato::ScalarVector a_x,
        Plato::ScalarVector a_b)
        {

            Plato::RowSum<NumDofsPerNode> rowSum(a_A);

            Plato::InverseWeight<NumDofsPerNode> inverseWeight;

            Plato::ScalarVector tRowSum("row sum", a_x.extent(0));

            // a_x[i] 1.0/sum_j(a_A[i,j]) * a_b[i]
            auto tNumBlockRows = a_A->rowMap().size() - 1;
            Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumBlockRows), LAMBDA_EXPRESSION(int blockRowOrdinal)
            {
                // compute row sum
                rowSum(blockRowOrdinal, tRowSum);

                // apply inverse weight
                inverseWeight(blockRowOrdinal, tRowSum, a_b, a_x, /*scale=*/-1.0);
                
            }, "row sum inverse");
        }

} // namespace Solve

} // namespace Plato

#endif
