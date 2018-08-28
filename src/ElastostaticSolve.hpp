
#ifndef ELASTOSTATICS_SOLVE_HPP
#define ELASTOSTATICS_SOLVE_HPP

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <CrsLinearProblem.hpp>
#include <CrsMatrix.hpp>
#include <Fields.hpp>
#include <ParallelComm.hpp>

#include "ErrorHandling.hpp"

#include "plato/PlatoStaticsTypes.hpp"

#include "ImplicitFunctors.hpp"
#include "LinearElasticMaterial.hpp"

namespace Plato {

namespace LinearElastostatics {


  /******************************************************************************/
  /*!
    \brief Linear static mechanics class.
  
    This class computes the global stiffness matrix and forcing vector
    that results from finite-element discretization of small strain elasticity:
    \f{eqnarray*}{
      \nabla\dot\sigma = f_b\\
      \sigma = C \epsilon\\
      \epsilon = \nabla_s u
    \f}
  */
  template<int SpatialDim>
  class ElastostaticSolve
  /******************************************************************************/
  {
    using DefaultFields = lgr::Fields<SpatialDim>;
  public:
    // 2D view for multiple RHSes
    typedef Kokkos::View<Plato::Scalar**, Kokkos::LayoutRight, Plato::MemSpace>    ScalarMultiVector;
    typedef Kokkos::View<Plato::Scalar***, Kokkos::LayoutRight, Plato::MemSpace>   CellWorkset;
    
    typedef lgr::CrsLinearProblem<Plato::OrdinalType> CrsLinearSolver;

  private:
    Teuchos::RCP<Plato::CrsMatrixType> m_matrix;
    Plato::ScalarVector       m_lhs, m_rhs;

    Plato::LocalOrdinalVector m_bcDofs;
    Plato::ScalarVector       m_bcValues;

    Teuchos::RCP<Plato::LinearElasticMaterial<SpatialDim>> m_materialModel;
    
  public:

    /*!
      \brief Computes the global stiffness matrix, global forcing vector, and applies
       constraints to both.
    */
    template <template<int Dimension, int NumDofsPerNode_I, int NumDofsPerNode_J> class LookupType>
    void computeGlobalStiffness();
    void assemble();
    void computeBodyForces();

  private:
    lgr::comm::Machine m_machine;
    Teuchos::RCP<DefaultFields> m_meshFields;

    const bool m_useBlockMatrix;
    
  public:
    ElastostaticSolve(Teuchos::ParameterList const& paramList,
                      Teuchos::RCP<DefaultFields> meshFields,
                      lgr::comm::Machine machine);

    /*!
      \brief Initialize the ElastostaticSolve object.

      Creates the global matrix, global solution vector, and global forcing 
      vector.  This function can be called multiple times.  It must be called before
      calling the assemble() function.
    */
    void initialize();
    
    /*!
      \brief Set Dirichlet constraints
    
      \param localDofOrdinals List of ordinals of constrained nodes.
      \param values List of values to which constrained nodes will be constrained.
    */
    void setBC(const Omega_h::LOs localNodeOrdinals, const Omega_h::Reals values, bool addToExisting = false);
    
    // ! Returns the constrained dofs ordinals
    decltype(m_bcDofs)& getConstrainedDofs(){ return m_bcDofs; }

    // ! Returns the constrained values
    decltype(m_bcValues)& getConstrainedValues(){ return m_bcValues; }

    // ! Returns the stiffness matrix as const
    const Plato::CrsMatrixType& getMatrix() const {return *m_matrix;}

    // ! Returns the stiffness matrix
    Plato::CrsMatrixType getMatrix() {return *m_matrix;}
    
    // ! returns the solution vector
    decltype(m_lhs)& getLHS() {return m_lhs;}
    
    // ! returns the RHS vector
    decltype(m_rhs)& getRHS() {return m_rhs;}
    
    Teuchos::RCP<CrsLinearSolver> getDefaultSolver(double tol, int maxIters);
  };

} // end namespace LinearElastostatics

} // end namespace Plato

#endif
