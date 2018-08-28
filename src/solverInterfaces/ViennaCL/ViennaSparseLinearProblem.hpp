//
//  ViennaSparseLinearProblem.hpp
//  
//
//  Created by Roberts, Nathan V on 8/8/17.
//
//

#ifndef ViennaSparseLinearProblem_h
#define ViennaSparseLinearProblem_h

#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>

#ifdef HAVE_MPI
#include <Teuchos_DefaultMpiComm.hpp>
#endif

#include <CrsLinearProblem.hpp>
//#include <KokkosKernels_SPGEMM_viennaCL_impl.hpp>

#include <viennacl/compressed_matrix.hpp>
#include <viennacl/linalg/cg.hpp>

namespace lgr {
  template<class Ordinal>
  class ViennaSparseLinearProblem : public CrsLinearProblem<Ordinal>
  {
  private:
    typedef Kokkos::View<Scalar*,  MemSpace>                 Vector;
    typedef Kokkos::View<Scalar**, Kokkos::LayoutRight, MemSpace>    MultiVector;
    typedef int                                                      RowMapType;
    typedef CrsMatrix<Ordinal, RowMapType> Matrix;
    
    typedef Ordinal LocalOrdinal;
    typedef Ordinal GlobalOrdinal;
    
    viennacl::compressed_matrix<Scalar> _matrix;
    
    Vector _x; // will want to copy here (from _lhs) in solve()...
    viennacl::vector<Scalar> _lhs, _rhs;
    
    int _maxIters = 1000;
    double _tol = 1e-10;
    
    int _iterationsTaken = -1;
    double _residualEstimate = -1.0;
    
  public:
    ViennaSparseLinearProblem(const Matrix A, MultiVector x, const MultiVector b) : ViennaSparseLinearProblem(A, Vector(Kokkos::subview(x, 0, Kokkos::ALL())), Vector(Kokkos::subview(b, 0, Kokkos::ALL())))
    {
      TEUCHOS_TEST_FOR_EXCEPTION(b.dimension_0() != 1, std::invalid_argument, "We do not yet support having multiple RHSes");
      TEUCHOS_TEST_FOR_EXCEPTION(x.dimension_0() != 1, std::invalid_argument, "We do not yet support having multiple RHSes");
    }
    
    ViennaSparseLinearProblem(const Matrix A, Vector x, const Vector b) : CrsLinearProblem<Ordinal>(A,x,b)
    {
      // everything below assumes exactly one MPI rank.  So, in fact, do the incoming arguments...
      
      _x = x;
      Ordinal N = x.size();
      
      /**** SET UP MATRIX ON DEVICE ****/
      // the following construction is sub-optimal (using STL map in serial on host, etc.)
      // ViennaCL's construction operators don't give us a great deal of flexibility, so until/unless we find this to be
      // a bottleneck, we'll just go with the simplest implementation, rather than trying to optimize
      std::vector< std::map< LocalOrdinal, Scalar> > cpu_sparse_matrix(N);
      
      typedef Kokkos::View<Ordinal*, Layout, MemSpace>     OrdinalVector;
      typedef Kokkos::View<RowMapType*, Layout, MemSpace>  RowMapTypeVector;
      
      typename RowMapTypeVector::HostMirror      rowMapHost = Kokkos::create_mirror_view( A.rowMap()        );
      typename OrdinalVector::HostMirror  columnIndicesHost = Kokkos::create_mirror_view( A.columnIndices() );
      typename Vector::HostMirror               entriesHost = Kokkos::create_mirror_view( A.entries()       );
      
      Kokkos::deep_copy( rowMapHost,        A.rowMap()        );
      Kokkos::deep_copy( columnIndicesHost, A.columnIndices() );
      Kokkos::deep_copy( entriesHost,       A.entries()       );
      
      using namespace std;
      
      RowMapType rowCount = rowMapHost.size() - 1;
//      cout << "rowCount = " << rowCount << endl;
//      cout << "b.size() = " << b.size() << endl;
      TEUCHOS_TEST_FOR_EXCEPTION(rowCount != RowMapType(N), std::invalid_argument, "matrix size and x length do not match");
      int entry = 0;
      for (RowMapType row=0; row<rowCount; row++)
      {
        int colsForRow = rowMapHost(row+1) - rowMapHost(row);
        for (int colOrdinal=0; colOrdinal<colsForRow; colOrdinal++)
        {
          int col = columnIndicesHost(entry);
          Scalar value = entriesHost(entry);
          cpu_sparse_matrix[row][col] = value;
//          cout << "A[" << row << "][" << col << "] = " << value << endl;
          entry++;
        }
      }
      
      //set up a sparse ViennaCL matrix:
      _matrix = viennacl::compressed_matrix<Scalar>(N, N);
      
//      cout << "entry = " << entry << endl;
//      cout << "N = " << N << endl;
      
//      cout << "_matrix has dimensions " << _matrix.size1() << " x " << _matrix.size2() << endl;
      
      //copy to device:
      viennacl::copy(cpu_sparse_matrix, _matrix);
      
      /**** SET UP VECTORS ON DEVICE ****/
      // Again, we're starting out with the sure, simple, but inefficient implementation
      std::vector<Scalar> cpu_rhs(N);
      std::vector<Scalar> cpu_lhs(N);
      
      typename Vector::HostMirror rhsHost = Kokkos::create_mirror_view( b );
      typename Vector::HostMirror lhsHost = Kokkos::create_mirror_view( x );
      
      Kokkos::deep_copy( rhsHost, b );
      Kokkos::deep_copy( lhsHost, x );
      
      for (int entryOrdinal=0; entryOrdinal<N; entryOrdinal++)
      {
        cpu_rhs[entryOrdinal] = rhsHost(entryOrdinal);
        cpu_lhs[entryOrdinal] = lhsHost(entryOrdinal);
      }
      
//      for (int entryOrdinal=0; entryOrdinal<N; entryOrdinal++)
//        cout << "x[" << entryOrdinal << "] = " << cpu_lhs[entryOrdinal] << endl;
//      for (int entryOrdinal=0; entryOrdinal<N; entryOrdinal++)
//        cout << "b[" << entryOrdinal << "] = " << cpu_rhs[entryOrdinal] << endl;
      
      _rhs.resize(N);
      _lhs.resize(N);
      
      //copy rhs,lhs to GPU vector
      viennacl::copy(cpu_rhs.begin(), cpu_rhs.end(), _rhs.begin());
      viennacl::copy(cpu_lhs.begin(), cpu_lhs.end(), _lhs.begin());
    }
    
    void initializePreconditioner() // TODO: add mechanism for setting options
    {
//      Teuchos::ParameterList params;
//      _preconditioner = MueLu::CreateViennaPreconditioner(Teuchos::rcp_static_cast<Operator>(_matrix), params);
    }
    
    void initializeSolver() // TODO: add mechanism for setting options
    {
      using namespace std;
      using namespace Teuchos;
      
//      RCP<ParameterList> belosPL = rcp( new ParameterList );
//      string solverName = "Pseudoblock CG"; // "Pseudoblock GMRES", "Klu2" are also available
//      belosPL->set("Solver Name",solverName);
//      
//      string residualScaling = "Norm of RHS";
////      string residualScaling = "Norm of Initial Residual";
////      string residualScaling = "Norm of Preconditioned Initial Residual";
////      string residualScaling = "None";
//      belosPL->set( "Implicit Residual Scaling", residualScaling );
//      
//      // Make Belos produce output like AztecOO's.
//      belosPL->set("Output Style", static_cast<int> (Belos::Brief));
//      // Always print Belos' errors.  You can add the enum values together to get all of their effects.
//      Belos::MsgType belosPrintOptions = static_cast<Belos::MsgType> (static_cast<int> (Belos::Errors) + static_cast<int> (Belos::Warnings));
//
////      // no output
////      belosPL->set("Output Frequency", -1);
////      
////      // maximum output
////      belosPL->set("Output Frequency", 1);
////      belosPrintOptions = static_cast<Belos::MsgType> (static_cast<int> (belosPrintOptions) + static_cast<int> (Belos::StatusTestDetails));
////      belosPrintOptions = static_cast<Belos::MsgType> (static_cast<int> (belosPrintOptions) + static_cast<int> (Belos::FinalSummary));
////      
////      // only print the final result
////      belosPL->set("Output Frequency", -1);
////      belosPrintOptions = static_cast<Belos::MsgType> (static_cast<int> (belosPrintOptions) + static_cast<int> (Belos::FinalSummary));
////      
//      // output every 10 iterations:
//      const int frequency = 10;
//      belosPL->set("Output Frequency", frequency);
//      belosPrintOptions = static_cast<Belos::MsgType> (static_cast<int> (belosPrintOptions) + static_cast<int> (Belos::StatusTestDetails));
//      belosPrintOptions = static_cast<Belos::MsgType> (static_cast<int> (belosPrintOptions) + static_cast<int> (Belos::FinalSummary));
//      
//      /*belosPL->set("Verbosity", static_cast<int> (belosPrintOptions));
//      // Only set the "Orthogonalization" parameter if using GMRES.  CG
//      // doesn't accept that parameter.
//      if (aztec_options[AZ_solver] == AZ_gmres) {
//        switch(aztec_options[AZ_orthog]) {
//          case AZ_classic:
//            belosPL->set("Orthogonalization","ICGS");
//            break;
//          case AZ_modified:
//            belosPL->set("Orthogonalization","IMGS");
//            break;
//          default:
//            // Belos doesn't support DGKS
//            error_string="Translate_Params_Aztec_to_Belos: unavailable AZ_orthog option";
//            return -6;
//        }
//      }
//      // Only set the "Num Blocks" (restart length) parameter if using
//      // GMRES.  CG doesn't accept that parameter.
//      if (aztec_options[AZ_solver] == AZ_gmres && aztec_options[AZ_kspace] !=0) {
//        belosPL->set("Num Blocks",aztec_options[AZ_kspace]);
//      }*/
//      
//      
//      int maxIters = 1000;
//      double tolerance = 1e-10;
//      
//      belosPL->set("Maximum Iterations", maxIters);
//      belosPL->set("Convergence Tolerance", tolerance);
//      
//      int numRHSes = 1;
//      belosPL->set("Deflation Quorum", numRHSes);
//      SolverFactory factory;
//      _solver = factory.create("CG", belosPL);
    }
    
    int getIterationsTaken()
    {
      return _iterationsTaken;
    }
    
    void setMaxIters(int maxCGIters)
    {
      _maxIters = maxCGIters;
    }
    
    double getResidual()
    {
      return _residualEstimate;
    }
    
    void setTolerance(double tol)
    {
      _tol = tol;
    }
    
    int solve() {
      using namespace std;
      
      // Set up CG solver object
      viennacl::linalg::cg_tag my_cg_tag(_tol, _maxIters);
      viennacl::linalg::cg_solver<viennacl::vector<Scalar> > solver(my_cg_tag);
      solver.set_initial_guess(_lhs);
      
      // TODO: add preconditioner (AMG)
      
//      cout << "A: matrix with dimensions " << _matrix.size1() << " x " << _matrix.size2() << endl;
//      cout << "lhs: vector of length " << _lhs.size() << endl;
//      cout << "rhs: vector of length " << _rhs.size() << endl;
      
      // solve:
      _lhs = solver(_matrix, _rhs);
      
      _iterationsTaken = solver.tag().iters();
      _residualEstimate = solver.tag().error();
      
      // copy from _lhs to _x (the Kokkos View)
      Ordinal N = _x.size();
      std::vector<Scalar> cpu_lhs(N);
      viennacl::copy(_lhs.begin(), _lhs.end(), cpu_lhs.begin());
      
      typename Vector::HostMirror xHost = Kokkos::create_mirror_view( _x );
      
      for (int entryOrdinal=0; entryOrdinal<N; entryOrdinal++)
      {
        xHost(entryOrdinal) = cpu_lhs[entryOrdinal];
      }

      // copy from host to device
      Kokkos::deep_copy( _x, xHost );
      
      return 0; // TODO: figure out how to get a result code from ViennaCL
    }
    
    ~ViennaSparseLinearProblem()
    {
      
    }
  };
}

#endif /* ViennaSparseLinearProblem_h */
