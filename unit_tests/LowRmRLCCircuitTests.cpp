#include "LGRTestHelpers.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_map.hpp"

#include "Teuchos_UnitTestHarness.hpp"

#include "CrsMatrix.hpp"
#include "FEMesh.hpp"
#include "LowRmRLCCircuit.hpp"
#include "MatrixIO.hpp"
#include "MeshFixture.hpp"

#include <sstream>

using namespace lgr;
using namespace Omega_h;
using namespace std;

namespace {
  void testFloatEqualsWithFloor(double v1, double v2, double tol, double floor, Teuchos::FancyOStream &out, bool &success)
  {
    if ((std::abs(v1) < floor) && (std::abs(v2) < floor)) return; // bypass relative comparison for things that floor to 0
    TEST_FLOATING_EQUALITY(v1, v2, tol);
  }
  
  TEUCHOS_UNIT_TEST( LowRmRLCCircuit, BackwardEulerSolve )
  {
    /*
     
     Solve A x = b, where
     
        [ C / dt                    1  ]
        [          G   -G          -1  ]
    A = [         -G  G+K11            ]
        [                     1        ]
        [  -1      1              L/dt ]
     
     and
     
    x = [v1, v2, v3, v4, i],
     
    b = [(C/dt)v1_{n-1} 0 0 0 (L/dt)i_{n-1}]
     
     */
    
    // coefficients dt, C, L, R, K11: (G = 1/R)
    vector<double> dt_values   = {1.0,0.1,0.01,0.001};
    vector<double> C_values    = {1.0, 2.0};
    vector<double> L_values    = {1.0, 2.0};
    vector<double> R_values    = {1.0, 2.0};
    vector<double> K11_values  = {1.0, 2.0};
    
    double V0 = 3.0;
    double v1 = V0, v2 = 0.0, v3 = 0.0, v4 = 0.0, i = 0.0;
    
    double tol = 1e-15, floor = 1e-15;
    
    int numTimeSteps = 3;
    for (auto dt : dt_values)
    {
      for (auto C : C_values)
      {
        for (auto L : L_values)
        {
          for (auto R : R_values)
          {
            LowRmRLCCircuit circuit(R, L, C, V0);
            circuit.setValues(v1,v2,v3,v4,i);
            for (auto K11 : K11_values)
            {
              out << "*********** R = " << R << "; L = " << L << "; C = " << C;
              out << "; K11 = " << K11 << "; dt = " << dt << " ***********" << endl;
              
              circuit.setK11(K11);
              double v1_prev, i_prev; //, v2_prev, v3_prev, v4_prev, i_prev;
              circuit.setValues(V0, 0.0, 0.0, 0.0, 0.0);
              out << "initial values: ";
              out << "v1 = " << circuit.v1() << "; ";
              out << "v2 = " << circuit.v2() << "; ";
              out << "v3 = " << circuit.v3() << "; ";
              out << "v4 = " << circuit.v4() << "; ";
              out << " i = " << circuit.i()  << "\n";
              
              for (int n=0; n<numTimeSteps; n++)
              {
                v1_prev = circuit.v1();
//                v2_prev = circuit.v2();
//                v3_prev = circuit.v3();
//                v4_prev = circuit.v4();
                i_prev  = circuit.i();
                circuit.takeTimeStep(dt);
                v1 = circuit.v1();
                v2 = circuit.v2();
                v3 = circuit.v3();
                v4 = circuit.v4();
                i  = circuit.i();
                out << "values after " << n+1 << " timesteps of size " << dt << endl;
                out << "v1 = " << circuit.v1() << "; ";
                out << "v2 = " << circuit.v2() << "; ";
                out << "v3 = " << circuit.v3() << "; ";
                out << "v4 = " << circuit.v4() << "; ";
                out << " i = " << circuit.i()  << "\n";
                
                // apply A to the new values to recover the previous values
                double v1_prev_actual = ((C / dt) * v1 + i) / (C / dt);
                double i_prev_actual  = (-v1 + v2 + (L / dt) * i) / (L / dt);
                testFloatEqualsWithFloor(v1_prev_actual, v1_prev, tol, floor, out, success);
                testFloatEqualsWithFloor( i_prev_actual,  i_prev, tol, floor, out, success);
              }
            }
          }
        }
      }
    }
    
  }
} // namespace
