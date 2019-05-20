#ifndef LGR_CIRCUIT_HPP
#define LGR_CIRCUIT_HPP

#include <Omega_h_input.hpp>
#include <lgr_linear_algebra.hpp>
#include<string>
#include<vector>

namespace lgr {

class Circuit
{
   private:
      // Node counts:
      int nNum; // Total number of nodes (includes grounds)
      int gNum; // Number of grounds

      // Element counts:
      int eNum; // Total number of elements
      int rNum; // Number of resistors
      int vNum; // Number of voltage sources
      int cNum; // Number of capacitors

      int gNumMax; // Max node number of ground nodes
      int nNumMin; // Minimum overall node number

      // Element type integers:
      const int ETYPE_RESISTOR = 1;
      const int ETYPE_CAPACITOR = 2;
      const int ETYPE_VSOURCE = 3;

      // Matrix information:
      //    The system being solved is:
      //
      //                   dx
      //       A * x + B * -- = r
      //                   dt
      //
      //    * Note that for x', we use a simple backward difference method by saving
      //      the previous time steps solution. This only comes into play from capacitors
      //
      //    * To start, initial voltage across capacitor is required. As a result, we initially
      //      solve:
      //
      //       A * x = r
      //
      //      where the inital specified voltage across the capacitors are treated as a voltage source
      //    
      //
      //    A[i][j]:
      //       * Conducatance matrix from resistors
      //       * Also stores voltage drop constraint when present
      //
      //       * 0 <= i < nNum - gNum:
      //          - i equation is a KCL equation for (i + nNumMin) node, excluding grounds
      //       * nNum - gNum <= i < NA:
      //          - i equation is a voltage drop equation constraint at node (i + nNumMin)
      //       * 0 <= j < nNum - gNum:
      //          - j corresponds to unknown voltage for (j + nNumMin) node, excluding grounds
      //       * nNum - gNum <= j < NA:
      //          - j corresponds to unknown current that flows through nodes (j + nNumMin) and (i + nNumMin)
      //    B[i][j]:
      //       * Capacitance matrix from capacitors
      //       * Does NOT store voltages; capacitors with given voltage are treated as voltage drop constraint in A
      //       * Similar ordering and interpretation as A[][]
      //    r[i]:
      //       * System right side forcing for (i + nNumMin) node
      //    x[j]:
      //       * System solution
      //       * 0 <= j < nNum - gNum: Voltage, due to KCL
      //       * nNum - gNum <= j < NA: Current, due to voltage constraint
      int NA;
      double **A, **B;  
      double *r, *x;

      // When solution is advanced in time, dt is used
      double dt;

      // When true or false, the second and first equations above are solved, respectuflly
      bool solveOnly;
      // When true, the pointer array memory will be created
      bool firstCall;

      // Element and node maps:
      //    i: element number (0 <= i < eNum)
      //    j: node number (0 <= i < 2)
      //
      // enMap[i][j]:
      //    * Given i element, gives nodes across it
      std::vector< std::vector<int> > enMap;
      //
      // eType[i]:
      //    * Given i element, gives what type it is
      std::vector<int> eType;
      //
      // eValMap[i]:
      //    * Given i element, what xVal is stored
      //    * Must check eType[i] first to see what element
      //      list to look at:
      //      
      //       l = eType[i]
      //       m = eValMap[i]
      //
      //       if l == ETYPE_RESISTOR:
      //          rVal[m] gives conductance of resistor
      //       if l == ETYPE_CAPACITOR:
      //          cVal[m] gives capacitance of capacitor
      //          v0Val[m][0] gives specified initial voltage at node enMap[i][0]
      //          v0Val[m][1] gives specified initial voltage at node enMap[i][1]
      //       if l == ETYPE_VSOURCE:
      //          vVal[m] gives initial voltage across voltage source
      std::vector<double> eValMap;
      std::vector<double> rVal;
      std::vector<double> cVal;
      std::vector<double> vVal;
      std::vector< std::vector<double> > v0Val;

      // gNodes[k]:
      //    * 0 <= k < gNum ground node number list
      std::vector<int> gNodes;

      std::vector<int> cVNodes;
      std::vector<int> cVVals;

      std::vector<int> icVNodes;
      std::vector<int> icVVals;

//    void Initialize(YAML::Node config);
      void Initialize(Omega_h::InputMap& pl);
      void AddType(std::string eTypein);
      void AddNodes(std::vector<int> &nodes);
      void AddConductance(double &data);
      void AddCapacitance(double &data);
      void AddVoltage(double &data);
      void AddCVoltage(std::vector<double> &data);
      void AddGround(double &data);
      void ComponentMap();
      void NodeCount();
      void AssembleMatrix();
      void SolveMatrix();
      void GaussElim(double **Ai, double *bio, int n);

//    void ParseYAML(YAML::Node config);
      void ParseYAML(Omega_h::InputMap& pl);

   public:

//    Circuit(YAML::Node config);
      Circuit();
      ~Circuit();
      void Setup(Omega_h::InputMap& pl);
      void SayInfo();
      void SayMatrix();
      void SayVoltages();

      void Solve(); // Initial solve without advance in time
      void Solve(double dtin); // Solve with advance in time
      double GetNodeVoltage(int nodein); // Measure voltage at user specified node
      void SetElementConductance(int e, double c); // Change resistor conductance of user specified element e to c 

      int GetNumNodes();
      int GetNumElements();
      int GetNumResistors();
      int GetNumCapacitors();
      int GetNumVSources();
      int GetNumGrounds();
};


// assembles M * udot + K * u = b
void assemble_circuit(std::vector<int> const& resistor_dofs,
    std::vector<int> const& inductor_dofs,
    std::vector<int> const& capacitor_dofs,
    std::vector<double> const& resistances,
    std::vector<double> const& inductances,
    std::vector<double> const& capacitances, int const ground_dof,
    MediumMatrix& M, MediumMatrix& K);

void form_backward_euler_circuit_system(MediumMatrix const& M,
    MediumMatrix const& K, MediumVector const& last_x, double const dt,
    MediumMatrix& A, MediumVector& b);

}  // namespace lgr

#endif
