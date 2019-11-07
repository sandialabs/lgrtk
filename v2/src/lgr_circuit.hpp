#ifndef LGR_CIRCUIT_HPP
#define LGR_CIRCUIT_HPP

#include<Omega_h_input.hpp>
#include<Omega_h_expr.hpp>
#include<lgr_linear_algebra.hpp>
#include<string>
#include<vector>

namespace lgr {

struct Branch {
    int count = 1;
    std::vector<int> elements;
};

class Circuit
{
   private:
      // Node counts
      int nNum;    // Total number of nodes (includes grounds)
      int gNum;    // Number of grounds (V = 0)
      int gNumMax; // Max node number of ground nodes
      int nNumMin; // Minimum overall node number

      // Element counts
      int eNum;  // Total number of elements
      int rNum;  // Number of resistors
      int cNum;  // Number of capacitors
      int lNum;  // Number of inductors
      int vNum;  // Number of constraints
      int eMesh; // Mesh element number

      // Time step to advance circuit by (L/C only)
      double dt;
      double time;

      // Logic switches
      bool firstCall;

      // Element types
      const int ETYPE_RESISTOR  = 1;
      const int ETYPE_CAPACITOR = 2;
      const int ETYPE_INDUCTOR  = 3;

      // Nearly zero number
      const double rthresh = 1E-12;

      // Full assembled matrix/vectors: (N + M) * x = NM * x = b
      int nmat_size;
      MediumMatrix N_matrix;
      MediumMatrix M_matrix;
      MediumMatrix NM_matrix;
      MediumVector b_vector;
      MediumVector x_vector;

      // Non-square matricies/vectors
      std::vector< std::vector<int> > AR_matrix;
      std::vector< std::vector<double> > G_matrix;
      std::vector< std::vector<int> > AC_matrix;
      std::vector< std::vector<double> > C_matrix;
      std::vector< std::vector<int> > AL_matrix;
      std::vector< std::vector<double> > L_matrix;
      std::vector< std::vector<int> > AV_matrix;
      std::vector<double> V_vector;
      std::vector<double> I_vector;

      // Various element/node maps
      std::vector< std::vector<int> > enMap; // Element to node
      std::vector<int> eType;                // Element type
      std::vector<int> eValMap;              // Element to value map
      std::vector<int> eNumMap;              // Element to user number map
      std::vector<double> rVal;              // Resistance values
      std::vector<double> cVal;              // Capacitance values
      std::vector<double> lVal;              // Inductance values

      // Fixed voltage and/or current nodes
      std::vector<double> fvVals; // fixed voltage values
      std::vector<int> fvNodes;   // fixed voltage nodes
      std::vector<double> fiVals; // fixed current values
      std::vector<int> fiNodes;   // fixed current nodes

      // Initial voltage and/or current nodes
      std::vector<double> ivVals; // initial voltage values
      std::vector<int> ivNodes;   // initial voltage nodes
      std::vector<double> iiVals; // initial current values
      std::vector<int> iiNodes;   // initial current nodes

      // Ground node list
      std::vector<int> gNodes;

      // Branch list
      std::vector<Branch> branch;

      // Specifying circuit
      void AddElement(std::string eTypein, int &e);
      void AddNodes(std::vector<int> &nodes);
      void AddConductance(double &data);
      void AddCapacitance(double &data);
      void AddInductance(double &data);

      // Setup routines
      template <typename T> T
      ConvertNodeToEq(int node);
      void ComponentMap();
      void NodeCount();
      void UpdateGrounds();
      void UpdateMatrixSize();
      void UpdateBranchValues();
      void ParseYAML(Omega_h::InputMap& pl, Omega_h::ExprEnv& env_in);
      int get_int(Omega_h::ExprEnv& env_in, std::string& expr);
      double get_double(Omega_h::ExprEnv& env_in, std::string& expr);

      // Solve routines
      void AssembleMatrix();
         void ZeroMatrix();
         void AssembleRMatrix();
         void AssembleCMatrix();
         void AssembleLMatrix();
         void AssembleSourceMatrix();
         void AssembleNMatrix();
         void ModifyCEquation();
         void AssembleMMatrix();
         void AssembleNMMatrix();
         void AssemblebVector();
      void SolveMatrix();


   public:
      // Logic switches
      bool usingCircuit;
      bool usingMesh;

      // Constructors
      Circuit();
      ~Circuit();

      // Typical user interaction with circuit routines
      void Setup(Omega_h::ExprEnv& env_in, Omega_h::InputMap& pl);
      void Solve(double dtin, double timein); 
      double GetNodeVoltage(int nodein); 
      void SetElementConductance(int e, double c); 
      void SetMeshConductance(double c); 
      double GetElementConductance(int e);
      double GetMeshAnodeVoltage(); 
      double GetMeshCathodeVoltage(); 
      double GetMeshVoltageDrop(); 
      double GetMeshConductance();
      double GetMeshCurrent();

      // Count Number of components
      int GetNumNodes();
      int GetNumElements();
      int GetNumResistors();
      int GetNumCapacitors();
      int GetNumInductors();
      int GetNumGrounds();

      // Routines below used for testing
      void Setup();
      void AddMeshUser     (int &e);
      void AddResistorUser (int &e, std::vector<int> &nodes, double &con);
      void AddCapacitorUser(int &e, std::vector<int> &nodes, double &cap);
      void AddInductorUser (int &e, std::vector<int> &nodes, double &ind);
      void AddFixedVUser   (std::vector<int> &nodes, std::vector<double> &vals);
      void AddInitialVUser (std::vector<int> &nodes, std::vector<double> &vals);
      void AddFixedIUser   (std::vector<int> &nodes, std::vector<double> &vals);
      void AddInitialIUser (std::vector<int> &nodes, std::vector<double> &vals);

      void SayInfo();
      void SayMatrix();
      void SayVoltages();
      void InitOutputMesh();
      void OutputMeshInfo();
};

}  // namespace lgr

#endif
