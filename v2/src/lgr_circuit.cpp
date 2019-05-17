#include <lgr_circuit.hpp>
#include <algorithm>
#include <iostream>

namespace lgr {

Circuit::Circuit()
{
   nNum    = 0;
   eNum    = 0;
   rNum    = 0;
   vNum    = 0;
   cNum    = 0;
   gNum    = 0;

   gNumMax = 0;
   nNumMin = 0;

   NA      = 0;

   solveOnly = true;
   firstCall = true;
}

void Circuit::Setup(Omega_h::InputMap pl)
{
   Initialize(pl);
   Solve();
}

Circuit::~Circuit()
{
   if (!firstCall) {
      for (int i=0; i<NA; i++) delete [] A[i];
      delete [] A;
      for (int i=0; i<NA; i++) delete [] B[i];
      delete [] B;
      
      delete [] r;
      delete [] x;
   }
}

double Circuit::GetNodeVoltage(int nodein)
{

   try {
      if (nodein < nNumMin || nodein >= nNumMin + nNum) {
         throw 1; // Invalid node number
      }
   }
   catch (int ex) {
      std::cout << "Circuit solve GetNodeVoltage() execption " << ex << std::endl;
   }
   int iflg = 0;
   for (int i=0; i<gNum; i++) {
      if (nodein == gNodes[i]) {
         iflg = 1;
         break;
      }
   }

   double voltage;
   if (iflg == 1) {
      voltage = 0.0; 
   } else {
      if (nodein >= gNumMax) nodein = nodein - gNum;
      voltage = x[nodein - nNumMin];
   }

  return voltage;
}

void Circuit::SetElementConductance(int e, double c)
{
   try {
      if (e >= eNum || e < 0) {
         throw 1; // Invalid element number
      } else if (eType[e] != ETYPE_RESISTOR) {
         throw 2; // Valid element number, but e is not resistor
      } else if (c <= 0) {
         throw 3; // Invalid capacitance
      }
   }
   catch (int ex) {
      std::cout << "Circuit solve SetElementConductance() execption " << ex << std::endl;
   }

   rVal[eValMap[e]] = c;
}

int Circuit::GetNumNodes()
{
   return nNum;
}

int Circuit::GetNumElements()
{
   return eNum;
}

int Circuit::GetNumResistors()
{
   return rNum;
}

int Circuit::GetNumCapacitors()
{
   return cNum;
}

int Circuit::GetNumVSources()
{
   return vNum;
}

int Circuit::GetNumGrounds()
{
   return gNum;
}

// void Circuit::Initialize(YAML::Node config)
void Circuit::Initialize(Omega_h::InputMap pl)
{
   ParseYAML(pl);
   SayInfo();
   ComponentMap();
   NodeCount();
}

void Circuit::Solve()
{
   AssembleMatrix();
   SolveMatrix();
}

void Circuit::Solve(double dtin)
{
   try {
      if (dtin <= 0) {
         throw 1; // Invalid time step
      }
   }
   catch (int ex) {
      std::cout << "Circuit solve Solve() execption " << ex << std::endl;
   }
   dt = dtin;

   solveOnly = false;
   Solve();
   solveOnly = true;
}

  
void Circuit::ParseYAML(Omega_h::InputMap pl)
{

   if (pl.is_map("circuit")) {
      // Alright, we do have a circuit in the yaml file
      auto& circuit_pl = pl.get_map("circuit");

      // Ground specification
      if (circuit_pl.is_list("ground nodes")) {
         auto& gNodes_pl = pl.get_list("ground nodes");
         for (int i=0; i < gNodes_pl.size(); i++) {
            gNodes[i] = gNodes_pl.get<int>(i);
         }
      }

   }


/*
   YAML::Node circuit = config["circuit"];

   // Grounds
   if (circuit["ground nodes"]){
      gNodes = circuit["ground nodes"].as<std::vector<int>>();
   }

   // Resistors
   YAML::Node element = circuit["resistors"];
   for (YAML::iterator it = element.begin(); it != element.end(); ++it) {
      std::string key = it->first.as<std::string>();
      if((key.compare("nodes")) == 0) {
         AddType("resistor");

         std::vector<int> nodes = it->second.as<std::vector<int>>();
         AddNodes(nodes);
      }
      if((key.compare("conductance")) == 0) {
         double data = it->second.as<double>();
         AddConductance(data);
      }
   }

   // Capacitors
   element = circuit["capacitors"];
   for (YAML::iterator it = element.begin(); it != element.end(); ++it) {
      std::string key = it->first.as<std::string>();
      if((key.compare("nodes")) == 0) {
         AddType("capacitor");

         std::vector<int> nodes = it->second.as<std::vector<int>>();
         AddNodes(nodes);
      }
      if((key.compare("capacitance")) == 0) {
         double data = it->second.as<double>();
         AddCapacitance(data);
      }
      if((key.compare("initial voltage")) == 0) {
         std::vector<double> data = it->second.as<std::vector<double>>();
         AddCVoltage(data);
      }
   }

   // Voltage sources
   element = circuit["vsources"];
   for (YAML::iterator it = element.begin(); it != element.end(); ++it) {
      std::string key = it->first.as<std::string>();
      if((key.compare("nodes")) == 0) {
         AddType("vsource");

         std::vector<int> nodes = it->second.as<std::vector<int>>();
         AddNodes(nodes);
      }
      if((key.compare("voltage")) == 0) {
         double data = it->second.as<double>();
         AddVoltage(data);
      }
   }
*/
}

void Circuit::AssembleMatrix()
{

   // Matrix sizes
   if (solveOnly) {
      NA = nNum - gNum + vNum + cNum; 
   } else {
      // Note that NA:NA+cNum are not being used this time around although they're there
      NA = nNum - gNum + vNum; 
   }

   // Setup the array memory
   if (firstCall) {
      if (NA > 0) {
         A = new double*[NA];
         for (int i=0; i<NA; i++) A[i] = new double[NA];
         B = new double*[NA];
         for (int i=0; i<NA; i++) B[i] = new double[NA];
         
         r  = new double[NA];
         x  = new double[NA];
         firstCall = false;
      }
   }

   for (int i=0; i<NA; i++) {
      for (int j=0; j<NA; j++) {
         A[i][j] = 0.0;
         B[i][j] = 0.0;
      }
      r[i] = 0;
   }


   int ii,jj,kk;
   int jva = -1;
   int node1,node2;
   int iflg1,iflg2;
   double vdrop;

   // Loop over elements
   for (int i=0; i<eNum; i++) {
      node1 = enMap[i][0];
      node2 = enMap[i][1];

      iflg1 = 0;
      iflg2 = 0;
      for (int j=0; j < gNum; j++) {
         if (node1 == gNodes[j]) iflg1 = 1;
         if (node2 == gNodes[j]) iflg2 = 1;
      }

      // KCL is first in matrix
      ii = node1 - nNumMin;
      jj = node2 - nNumMin;
      if (ii >= gNumMax - nNumMin) ii = ii - gNum;
      if (jj >= gNumMax - nNumMin) jj = jj - gNum;

      if (eType[i] == ETYPE_RESISTOR) {
         if ((iflg1 == 0) & (iflg2 == 0)) {
            // Myself equation ii
            A[ii][ii] = A[ii][ii] + rVal[eValMap[i]];
            A[ii][jj] = A[ii][jj] - rVal[eValMap[i]];

            // Neighbor equation jj
            A[jj][ii] = A[jj][ii] - rVal[eValMap[i]];
            A[jj][jj] = A[jj][jj] + rVal[eValMap[i]];
         } else if (iflg2 == 1) {
            // Myself equation ii only
            A[ii][ii] = A[ii][ii] + rVal[eValMap[i]];
         } else if (iflg1 == 1) {
            // Neighbor equation jj only
            A[jj][jj] = A[jj][jj] + rVal[eValMap[i]];
         }
      } else if (eType[i] == ETYPE_CAPACITOR) {
         if ((iflg1 == 0) & (iflg2 == 0)) {
            // Myself equation ii
            B[ii][ii] = B[ii][ii] + cVal[eValMap[i]];
            B[ii][jj] = B[ii][jj] - cVal[eValMap[i]];

            // Neighbor equation jj
            B[jj][ii] = B[jj][ii] - cVal[eValMap[i]];
            B[jj][jj] = B[jj][jj] + cVal[eValMap[i]];
         } else if (iflg2 == 1) {
            // Myself equation ii only
            B[ii][ii] = B[ii][ii] + cVal[eValMap[i]];
         } else if (iflg1 == 1) {
            // Neighbor equation jj only
            B[jj][jj] = B[jj][jj] + cVal[eValMap[i]];
         }

         // Treat capacitor as voltage source initially
         if (solveOnly) {
            jva++;
            kk = nNum - gNum + jva;
   
            if ((iflg1 == 0) & (iflg2 == 0)) {
               // Contribution to KCL at nodes
               A[ii][kk] = 1.0;
               A[jj][kk] = 1.0;
   
               // Voltage drop constraint
               A[kk][ii] = -1.0;
               A[kk][jj] = 1.0;

               vdrop = v0Val[eValMap[i]][1] - v0Val[eValMap[i]][0];
            } else if (iflg2 == 1) {
               // Contribution to KCL at nodes (non ground)
               A[ii][kk] = 1.0;
   
               // Voltage drop constraint
               A[kk][ii] = -1.0;

               vdrop = 0.0 - v0Val[eValMap[i]][0];
            } else if (iflg1 == 1) {
               // Contribution to KCL at nodes (non ground)
               A[jj][kk] = 1.0;
   
               // Voltage drop constraint
               A[kk][jj] = 1.0;

               vdrop = v0Val[eValMap[i]][1] - 0.0;
            }
   
            // Voltage drop constraint
            r[kk] = r[kk] + vdrop;

         }

      } else if (eType[i] == ETYPE_VSOURCE) {
         jva++;
         kk = nNum - gNum + jva;

         if ((iflg1 == 0) & (iflg2 == 0)) {
            // Contribution to KCL at nodes
            A[ii][kk] = 1.0;
            A[jj][kk] = 1.0;

            // Voltage drop constraint
            A[kk][ii] = -1.0;
            A[kk][jj] = 1.0;
         } else if (iflg2 == 1) {
            // Contribution to KCL at nodes (non ground)
            A[ii][kk] = 1.0;

            // Voltage drop constraint
            A[kk][ii] = -1.0;
         } else if (iflg1 == 1) {
            // Contribution to KCL at nodes (non ground)
            A[jj][kk] = 1.0;

            // Voltage drop constraint
            A[kk][jj] = 1.0;
         }

         // Voltage drop constraint
         r[kk] = r[kk] + vVal[eValMap[i]];
      }
   }
}

void Circuit::SolveMatrix()
{
  
   if (!solveOnly) {
      // A + alpha*B
      for (int i=0; i<NA; i++) {
      for (int j=0; j<NA; j++) {
         A[i][j] = A[i][j] + 1.0/dt*B[i][j];
      }
      }
      
      // RHS = r - alpha*B*x_{i-1}
      for (int i=0; i<NA; i++) {
      for (int j=0; j<NA; j++) {
         r[i] = r[i] + 1.0/dt*B[i][j]*x[j];
      }
      }
   } 

   // Also, we will copy r to x since x is altered on return
   for (int i=0; i<NA; i++) x[i] = r[i];

   // Gaussian Elimination
   GaussElim(A,x,NA);
}

void Circuit::GaussElim(double **Ain, double *bin, int n) {

    int i,j,k;

    double **tempmatrix =  new double*[n];
    for (i=0; i<n+1; i++) tempmatrix[i] = new double[n];
    double **aug = new double*[n];
    for (i=0; i<n+1; i++) aug[i] = new double[n];

    for (j=0; j<n; j++) {
    for (i=0; i<n; i++) {
       aug[i][j] = Ain[i][j];
       tempmatrix[i][j] = Ain[i][j];
    }
       aug[j][n] = bin[j];
       tempmatrix[j][n] = bin[j];

       bin[j] = 0; // also set b to x, so must be init to zero (in-place)
    }

    double maxx;
    int kc,kca;
    double rthresh = 1E-6;

    for (i=2;i<=n;i++) {

        // Find max of row
        maxx = tempmatrix[0][0];
        for (k=1;k<=n+1;k++) {
            if (tempmatrix[0][k-1] > maxx) maxx = tempmatrix[0][k-1];
        }

        // Divide row by that max
        for (k=1;k<=n+1;k++) tempmatrix[0][k-1] = tempmatrix[0][k-1]/maxx;

        // Find the maximum in a column
        maxx = abs(tempmatrix[0][0]);
        for (k=1;k<=n;k++) if (abs(tempmatrix[k-1][0]) > maxx) maxx = abs(tempmatrix[k-1][0]);

        std::vector<int> temp;
        kc  = 0;
        kca = 0;
        for (j=1;j<=n+1;j++){
            for (k=1;k<=n;k++){
                kca++;
                if (abs(tempmatrix[k-1][j-1] - maxx) > rthresh) {
                    kc++;
                    temp.push_back(kca);
                }
            }
        }

        int maxi = 1;
        if (kc > 2) {
           for (j=1;j<=kc-1;j++){
               if (j !=temp[j-1]) {
                   maxi = j;
                   break;
               }
           }
        } 

        // Row swap if maxi is not 1
        std::vector<double> temprow(n+1);
        if (maxi != 1) {
           for (j=1;j<=n+1;j++) {
               temprow[j-1] = tempmatrix[maxi-1][j-1];
               tempmatrix[maxi-1][j-1] = tempmatrix[0][j-1];
               tempmatrix[0][j-1] = temprow[j-1];
           }
        }

        // Row reducing
        double rsave;
        for (j=2;j<=n;j++) {
            rsave = tempmatrix[j-1][0];
            for (k=1;k<=n+1;k++) tempmatrix[j-1][k-1] = tempmatrix[j-1][k-1] - rsave/tempmatrix[0][0]*tempmatrix[0][k-1];
        }

        int jc = 0;
        for (j=i-1;j<=n;j++) {
            kc = 0;
            jc++;
        for (k=i-1;k<=n+1;k++) {
            kc++;
            aug[j-1][k-1] = tempmatrix[jc-1][kc-1];
        }
        }

        for (j=2;j<=n;j++) {
            for (k=2;k<=n+1;k++) tempmatrix[j-2][k-2] = tempmatrix[j-1][k-1];
        }
    }

    // Backward subsitute
    double dsum;
    bin[n-1] = aug[n-1][n]/aug[n-1][n-1];
    for (i=n-1;i>=1; i--) {
       dsum = 0;
       for (j=1;j<=n;j++) dsum = dsum + aug[i-1][j-1]*bin[j-1];
       bin[i-1] = (aug[i-1][n]-dsum)/aug[i-1][i-1];
    }

    // Clean up
    for (i=0; i<n+1; i++) delete [] tempmatrix[i];
    delete [] tempmatrix;
    for (i=0; i<n+1; i++) delete [] aug[i];
    delete [] aug;
}

void Circuit::AddType(std::string eTypein)
{
   if (eTypein.compare("resistor") == 0) {
      rNum++;
      eType.push_back(ETYPE_RESISTOR);
   } else if (eTypein.compare("vsource") == 0) {
      vNum++;
      eType.push_back(ETYPE_VSOURCE);
   } else if (eTypein.compare("capacitor") == 0) {
      cNum++;
      eType.push_back(ETYPE_CAPACITOR);
   }
}

void Circuit::ComponentMap()
{
   int ir = -1;
   int ic = -1;
   int iv = -1;
   for (int i=0; i<eNum; i++){
      if (eType[i] == ETYPE_RESISTOR) {
         ir++;
         eValMap.push_back(ir);
      } else if (eType[i] == ETYPE_CAPACITOR) {
         ic++;
         eValMap.push_back(ic);
      } else if (eType[i] == ETYPE_VSOURCE) {
         iv++;
         eValMap.push_back(iv);
      }
   }
}

void Circuit::NodeCount()
{
   // Count unique nodes as an O( elm^2 ) opperation
   int eNum2 = 2*eNum;
   int i1,i2,j1,j2,node1,node2,iflg;
   nNumMin = enMap[0][0];
   for (int i=0  ; i<eNum2; i++){
      i1 = i % eNum;
      i2 = i / 2;
      node1 = enMap[i1][i2];
      iflg = 0;

      // Also compute min node at same time
      if (node1 < nNumMin) nNumMin = node1;
   for (int j=i+1; j<eNum2; j++){
      j1 = j % eNum;
      j2 = j / 2;
      node2 = enMap[j1][j2];
      if (node1 == node2) {
         iflg = 1;
         break;
      }
   }
   if (iflg == 0) nNum++;
   }

   // Count ground nodes
   gNum = gNodes.size();
   if (gNum == 0) {
      gNumMax = nNum;
   } else {
      for (int i=0; i<gNum; i++){
         if (gNodes[i] > gNumMax) gNumMax = gNodes[i];
      }
   }

}

void Circuit::AddNodes(std::vector<int> &nodes)
{
   // Increment elements
   eNum++;

   // Add node pair for element
   enMap.push_back(nodes);
}

void Circuit::AddConductance(double &data)
{
   rVal.push_back(data);
}

void Circuit::AddCapacitance(double &data)
{
   cVal.push_back(data);
}

void Circuit::AddCVoltage(std::vector<double> &data)
{
   v0Val.push_back(data);
}

void Circuit::AddVoltage(double &data)
{
   vVal.push_back(data);
}

void Circuit::SayInfo()
{
   std::cout << "******************************" << std::endl;
   std::cout << "**** CIRCUIT INFORMATION: ****" << std::endl;
   std::cout << "******************************" << std::endl;
   std::cout << "Total nodes: "         << GetNumNodes()      << std::endl;
   std::cout << "Total ground nodes: "  << GetNumGrounds()    << std::endl;
   std::cout << "Total elements: "      << GetNumElements()   << std::endl;
   std::cout << std::endl;
   std::cout << "Resistors: "           << GetNumResistors()  << std::endl;
   std::cout << "Capacitors: "          << GetNumCapacitors() << std::endl;
   std::cout << "Voltage Sources: "     << GetNumVSources()   << std::endl;
   std::cout << std::endl;
}

void Circuit::SayVoltages()
{

   std::cout << "********************************" << std::endl;
   std::cout << "**** CIRCUIT NODE VOLTAGES: ****" << std::endl;
   std::cout << "********************************" << std::endl;

   double voltage;
   for (int i=nNumMin; i<nNum+nNumMin; i++){
      voltage = GetNodeVoltage(i);
      std::cout << "Voltage at node " << i << " is " << voltage << " V" << std::endl;
   }

   int node1,node2;
   for (int i=0; i<eNum; i++){
      node1 = enMap[i][0]; 
      node2 = enMap[i][1]; 
      voltage = GetNodeVoltage(node2) - GetNodeVoltage(node1);

      if (eType[i] == ETYPE_RESISTOR) {
         std::cout << "Voltage across " << 1.0/rVal[eValMap[i]] << " ohm resistor, between nodes " << node1 << " and " << node2 << " is " << voltage << " V" << std::endl;
      } else if (eType[i] == ETYPE_CAPACITOR) {
         std::cout << "Voltage across " << cVal[eValMap[i]] << " farad capacitor, between nodes " << node1 << " and " << node2 << " is " << voltage << " V" << std::endl;
      } else if (eType[i] == ETYPE_VSOURCE) {
         std::cout << "Voltage across " << vVal[eValMap[i]] << " V voltage source, between nodes " << node1 << " and " << node2 << " is " << voltage << " V" << std::endl;
      }
   }

   std::cout << std::endl;
}

void Circuit::SayMatrix()
{

   std::cout << "*************************************" << std::endl;
   std::cout << "**** CIRCUIT MATRIX INFORMATION: ****" << std::endl;
   std::cout << "*************************************" << std::endl;

   std::cout << "A: " << std::endl;
   for (int i=0; i<NA; i++){
      for (int j=0; j<NA; j++) std::cout << A[i][j] << " ";
      std::cout << std::endl;
   }
   std::cout << std::endl;

   std::cout << "B: " << std::endl;
   for (int i=0; i<NA; i++){
      for (int j=0; j<NA; j++) std::cout << B[i][j] << " ";
      std::cout << std::endl;
   }
   std::cout << std::endl;

   std::cout << "r: " << std::endl;
   for (int i=0; i<NA; i++){
      std::cout << r[i] << std::endl;
   }
   std::cout << std::endl;

   std::cout << "x: " << std::endl;
   for (int i=0; i<NA; i++){
      std::cout << x[i] << std::endl;
   }
   std::cout << std::endl;

   std::cout << std::endl;
}


void assemble_circuit(std::vector<int> const& resistor_dofs,
    std::vector<int> const& inductor_dofs,
    std::vector<int> const& capacitor_dofs,
    std::vector<double> const& resistances,
    std::vector<double> const& inductances,
    std::vector<double> const& capacitances, int const ground_dof,
    MediumMatrix& M, MediumMatrix& K) {
  auto const nresistors = int(resistances.size());
  auto const ninductors = int(inductances.size());
  auto const ncapacitors = int(capacitances.size());
  OMEGA_H_CHECK(int(resistor_dofs.size()) == nresistors * 2);
  OMEGA_H_CHECK(int(inductor_dofs.size()) == ninductors * 3);
  OMEGA_H_CHECK(int(capacitor_dofs.size()) == ncapacitors * 2);
  int max_dof = -1;
  if (!resistor_dofs.empty())
    max_dof = std::max(
        max_dof, *std::max_element(resistor_dofs.begin(), resistor_dofs.end()));
  if (!inductor_dofs.empty())
    max_dof = std::max(
        max_dof, *std::max_element(inductor_dofs.begin(), inductor_dofs.end()));
  if (!capacitor_dofs.empty())
    max_dof = std::max(max_dof,
        *std::max_element(capacitor_dofs.begin(), capacitor_dofs.end()));
  int n = max_dof + 1;
  M = MediumMatrix(n);
  K = MediumMatrix(n);
  for (int c = 0; c < nresistors; ++c) {
    auto const i = resistor_dofs[std::size_t(c * 2 + 0)];
    auto const j = resistor_dofs[std::size_t(c * 2 + 1)];
    auto const R = resistances[std::size_t(c)];
    auto const G = 1.0 / R;
    K(i, i) += G * 1.0;
    K(j, j) += G * 1.0;
    K(i, j) += G * -1.0;
    K(j, i) += G * -1.0;
  }
  for (int c = 0; c < ninductors; ++c) {
    auto const i = inductor_dofs[std::size_t(c * 3 + 0)];
    auto const j = inductor_dofs[std::size_t(c * 3 + 1)];
    auto const k = inductor_dofs[std::size_t(c * 3 + 2)];
    auto const L = inductances[std::size_t(c)];
    K(i, k) += 1.0;
    K(j, k) += -1.0;
    K(k, i) += -1.0;
    K(k, j) += 1.0;
    M(k, k) += L;
  }
  for (int c = 0; c < ncapacitors; ++c) {
    auto const i = capacitor_dofs[std::size_t(c * 2 + 0)];
    auto const j = capacitor_dofs[std::size_t(c * 2 + 1)];
    auto const C = capacitances[std::size_t(c)];
    M(i, i) += C * 1.0;
    M(j, j) += C * 1.0;
    M(i, j) += C * -1.0;
    M(j, i) += C * -1.0;
  }
  for (int j = 0; j < n; ++j) {
    M(ground_dof, j) = (ground_dof == j) ? 1.0 : 0.0;
    K(ground_dof, j) = (ground_dof == j) ? 1.0 : 0.0;
  }
}

void form_backward_euler_circuit_system(MediumMatrix const& M,
    MediumMatrix const& K, MediumVector const& last_x, double const dt,
    MediumMatrix& A, MediumVector& b) {
  auto const n = M.size;
  A = MediumMatrix(n);
  b = MediumVector(n);
  auto const inv_dt = 1.0 / dt;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A(i, j) = inv_dt * M(i, j) + K(i, j);
      b(i) += inv_dt * M(i, j) * last_x(j);
    }
  }
}

}  // namespace lgr
