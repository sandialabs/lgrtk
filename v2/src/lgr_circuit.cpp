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

   eMesh = -1; // < 0 means mesh isn't set yet

   NA      = 0;

   solveOnly = true;
   firstCall = true;
}

Circuit::~Circuit()
{
}

void Circuit::AddResistorUser(int &e, std::vector<int> &nodes, double &con) {

   AddElement("resistor",e);
   AddNodes(nodes);
   AddConductance(con);
}

void Circuit::AddCapacitorUser(int &e, std::vector<int> &nodes, double &cap, std::vector<double> &v0) {

   AddElement("capacitor",e);
   AddNodes(nodes);
   AddCapacitance(cap);
   AddCVoltage(v0);
}

void Circuit::AddVSourceUser(int &e, std::vector<int> &nodes, std::vector<double> &v) {

   AddElement("vsource",e);
   AddNodes(nodes);
   AddVoltage(v);
}

void Circuit::AddGroundsUser(std::vector<int> &nodes) {

   int nadd = nodes.size();
   for (int i=0;i<nadd;i++) {
      gNodes.push_back(nodes[i]);
   }
}

void Circuit::AddMeshUser(int &e) {
   eMesh = e;
}

void Circuit::Setup(Omega_h::InputMap& pl)
{
   ParseYAML(pl);
   Setup();
}

void Circuit::Setup()
{
   Initialize();
   Solve();
   SayVoltages();

   for (int i=0;i<100;i++){
      Solve(0.01);
   }
   SayVoltages();
}

double Circuit::GetMeshAnodeVoltage()
{
   int efind = -1;
   try {
      for (int i=0;i<eNum;i++) {
         if (eNumMap[i] == eMesh) {
            efind = i;
            break;
         }
      }
      if (efind < 0) {
         throw 1; // Invalid mesh element number
      } else if (eType[efind] != ETYPE_RESISTOR) {
         throw 2; // Valid element number, but e is not resistor and thus not a mesh
      } 
   }
   catch (int ex) {
      std::cout << "Circuit solve GetMeshAnodeVoltage() execption " << ex << std::endl;
   }

   return GetNodeVoltage(enMap[efind][1]);
}

double Circuit::GetMeshCathodeVoltage()
{
   int efind = -1;
   try {
      for (int i=0;i<eNum;i++) {
         if (eNumMap[i] == eMesh) {
            efind = i;
            break;
         }
      }
      if (efind < 0) {
         throw 1; // Invalid mesh element number
      } else if (eType[efind] != ETYPE_RESISTOR) {
         throw 2; // Valid element number, but e is not resistor and thus not a mesh
      }
   }
   catch (int ex) {
      std::cout << "Circuit solve GetMeshCathodVoltage() execption " << ex << std::endl;
   }

   return GetNodeVoltage(enMap[efind][0]);
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
      voltage = x(nodein - nNumMin);
   }

  return voltage;
}

void Circuit::SetElementConductance(int e, double c)
{
   int efind = -1;
   try {
      for (int i=0;i<eNum;i++) {
         if (eNumMap[i] == e) {
            efind = i;
            break;
         }
      }
      if (efind < 0) {
         throw 1; // Invalid element number
      } else if (eType[efind] != ETYPE_RESISTOR) {
         throw 2; // Valid element number, but e is not resistor
      } else if (c <= 0) {
         throw 3; // Invalid capacitance
      }
   }
   catch (int ex) {
      std::cout << "Circuit solve SetElementConductance() execption " << ex << std::endl;
   }

   rVal[eValMap[efind]] = c;
}

void Circuit::SetMeshConductance(double c)
{
   SetElementConductance(eMesh, c);
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

void Circuit::Initialize()
{
   // firstCall means we can't add components but
   // only edit them after first Setup()
   if ((eNum > 0) && (firstCall)) {
      ComponentMap();
      NodeCount();
   }
   SayInfo();
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

  
void Circuit::ParseYAML(Omega_h::InputMap& pl)
{

   if (pl.is_map("circuit")) {
      // Alright, we do have a circuit in the yaml file
      auto& circuit_pl = pl.get_map("circuit");

      // Ground specification
      if (circuit_pl.is_list("ground nodes")) {
         auto& gNodes_pl = circuit_pl.get_list("ground nodes");
         for (int i=0; i < gNodes_pl.size(); i++) {
            gNodes.push_back(gNodes_pl.get<int>(i));
         }
      } // gNodes_pl

      // Mesh element number
      if (circuit_pl.is<int>("mesh element")) {
         eMesh = circuit_pl.get<int>("mesh element");
      } // Mesh element number

      // Resistors
      if (circuit_pl.is_list("resistors")) {
         auto& resistors_pl = circuit_pl.get_list("resistors");

         for (int i=0; i < resistors_pl.size(); i++) {
            if (resistors_pl.is_map(i)) {
                auto& rmap_pl = resistors_pl.get_map(i);

                // Add element number
                if (rmap_pl.is<int>("element")) {
                   int e = rmap_pl.get<int>("element");
                   AddElement("resistor",e);
                }

                // Add resistor nodes
                if (rmap_pl.is_list("nodes")){
                
                   auto& rnodes_pl = rmap_pl.get_list("nodes");
                   std::vector<int> rnodes;
                   for (int j=0; j < rnodes_pl.size(); j++) {
                      rnodes.push_back(rnodes_pl.get<int>(j));
                   }

                   AddNodes(rnodes);
                } 

                // Add resistor conductance
                if (rmap_pl.is<double>("conductance")) {
                   double rcval = rmap_pl.get<double>("conductance");
                   AddConductance(rcval);
                } 
                if (rmap_pl.is<int>("conductance")) {
                   double rcval = static_cast<double>(rmap_pl.get<int>("conductance"));
                   AddConductance(rcval);
                }
            }
         }
      } // resistors

      // Capacitors
      if (circuit_pl.is_list("capacitors")) {
         auto& capacitors_pl = circuit_pl.get_list("capacitors");

         for (int i=0; i < capacitors_pl.size(); i++) {
            if (capacitors_pl.is_map(i)) {
                auto& cmap_pl = capacitors_pl.get_map(i);

                // Add element number
                if (cmap_pl.is<int>("element")) {
                   int e = cmap_pl.get<int>("element");
                   AddElement("capacitor",e);
                }

                // Add capacitor nodes
                if (cmap_pl.is_list("nodes")){
                
                   auto& cnodes_pl = cmap_pl.get_list("nodes");
                   std::vector<int> cnodes;
                   for (int j=0; j < cnodes_pl.size(); j++) {
                      cnodes.push_back(cnodes_pl.get<int>(j));
                   }
                
                   AddNodes(cnodes);
                } 

                // Add capacitor capacitance
                if (cmap_pl.is<double>("capacitance")) {
                   double ccval = cmap_pl.get<double>("capacitance");
                   AddCapacitance(ccval);
                } 
                if (cmap_pl.is<int>("capacitance")) {
                   double ccval = static_cast<double>(cmap_pl.get<int>("capacitance"));
                   AddCapacitance(ccval);
                }

                // Add capacitor initial voltage
                if (cmap_pl.is_list("initial voltage")){
                
                   auto& cvnodes_pl = cmap_pl.get_list("initial voltage");
                   std::vector<double> cvnodes;
                   for (int j=0; j < cvnodes_pl.size(); j++) {
                      if (cvnodes_pl.is<double>(j)) {
                         cvnodes.push_back(cvnodes_pl.get<double>(j));
                      } else if (cvnodes_pl.is<int>(j)) {
                         int val = cvnodes_pl.get<int>(j);
                         double rval = 1.0*val;
                         cvnodes.push_back(rval);
                      }
                   }
                
                   AddCVoltage(cvnodes);
                } 
            }
         }
      } // capacitors

      // Voltage sources
      if (circuit_pl.is_list("voltage sources")) {
         auto& vsources_pl = circuit_pl.get_list("voltage sources");

         for (int i=0; i < vsources_pl.size(); i++) {
            if (vsources_pl.is_map(i)) {
                auto& vmap_pl = vsources_pl.get_map(i);

                // Add element number
                if (vmap_pl.is<int>("element")) {
                   int e = vmap_pl.get<int>("element");
                   AddElement("vsource",e);
                }

                // Add voltage nodes
                if (vmap_pl.is_list("nodes")){
                
                   auto& vnodes_pl = vmap_pl.get_list("nodes");
                   std::vector<int> vnodes;
                   for (int j=0; j < vnodes_pl.size(); j++) {
                      vnodes.push_back(vnodes_pl.get<int>(j));
                   }
                
                   AddNodes(vnodes);
                } 

                // Add voltage source voltage
                if (vmap_pl.is_list("voltage")){
                
                   auto& vvnodes_pl = vmap_pl.get_list("voltage");
                   std::vector<double> vvnodes;
                   for (int j=0; j < vvnodes_pl.size(); j++) {
                      if (vvnodes_pl.is<double>(j)) {
                         vvnodes.push_back(vvnodes_pl.get<double>(j));
                      } else if (vvnodes_pl.is<int>(j)) {
                         int val = vvnodes_pl.get<int>(j);
                         double rval = 1.0*val;
                         vvnodes.push_back(rval);
                      }
                   }
                
                   AddVoltage(vvnodes);
                } 
            }
         }
      } // voltage sources

   } // Circuit
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

   if (NA > 0) {

   // Setup the array memory
   if (firstCall) {
      A = MediumMatrix(NA);
      B = MediumMatrix(NA);
      r = MediumVector(NA);
      x = MediumVector(NA);

      firstCall = false;
   }

   for (int i=0; i<NA; i++) {
      for (int j=0; j<NA; j++) {
         A(i, j) = 0.0;
         B(i, j) = 0.0;
      }
      r(i) = 0;
   }


   int ii,jj,kk;
   int jva = -1;
   int node1,node2;
   int iflg1,iflg2;
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
            A(ii, ii) += rVal[eValMap[i]];
            A(ii, jj) -= rVal[eValMap[i]];

            // Neighbor equation jj
            A(jj, ii) -= rVal[eValMap[i]];
            A(jj, jj) += rVal[eValMap[i]];
         } else if (iflg2 == 1) {
            // Myself equation ii only
            A(ii, ii) += rVal[eValMap[i]];
         } else if (iflg1 == 1) {
            // Neighbor equation jj only
            A(jj, jj) += rVal[eValMap[i]];
         }
      } else if (eType[i] == ETYPE_CAPACITOR) {
         if ((iflg1 == 0) & (iflg2 == 0)) {
            // Myself equation ii
            B(ii, ii) += cVal[eValMap[i]];
            B(ii, jj) -= cVal[eValMap[i]];

            // Neighbor equation jj
            B(jj, ii) -= cVal[eValMap[i]];
            B(jj, jj) += cVal[eValMap[i]];
         } else if (iflg2 == 1) {
            // Myself equation ii only
            B(ii, ii) += cVal[eValMap[i]];
         } else if (iflg1 == 1) {
            // Neighbor equation jj only
            B(jj, jj) += cVal[eValMap[i]];
         }

         // Treat capacitor as voltage source initially
         if (solveOnly) {
            jva++;
            kk = nNum - gNum + jva;
   
            if ((iflg1 == 0) & (iflg2 == 0)) {
               // Contribution to KCL at nodes
               A(ii, kk) = 1.0;
               A(jj, kk) = 1.0;
   
               // Voltage drop constraint
               A(kk, ii) = -1.0;
               A(kk, jj) = 1.0;
            } else if (iflg2 == 1) {
               // Contribution to KCL at nodes (non ground)
               A(ii, kk) = 1.0;
   
               // Voltage drop constraint
               A(kk, ii) = -1.0;
            } else if (iflg1 == 1) {
               // Contribution to KCL at nodes (non ground)
               A(jj, kk) = 1.0;
   
               // Voltage drop constraint
               A(kk, jj) = 1.0;
            }
   
            // Voltage drop constraint
            r(kk) = r(kk) + (v0Val[eValMap[i]][1]-v0Val[eValMap[i]][0]); 

         }

      } else if (eType[i] == ETYPE_VSOURCE) {
         jva++;
         kk = nNum - gNum + jva;

         if ((iflg1 == 0) & (iflg2 == 0)) {
            // Contribution to KCL at nodes
            A(ii, kk) = 1.0;
            A(jj, kk) = 1.0;

            // Voltage drop constraint
            A(kk, ii) = -1.0;
            A(kk, jj) = 1.0;
         } else if (iflg2 == 1) {
            // Contribution to KCL at nodes (non ground)
            A(ii, kk) = 1.0;

            // Voltage drop constraint
            A(kk, ii) = -1.0;
         } else if (iflg1 == 1) {
            // Contribution to KCL at nodes (non ground)
            A(jj, kk) = 1.0;

            // Voltage drop constraint
            A(kk, jj) = 1.0;
         }

         // Voltage drop constraint
         r(kk) = r(kk) + (vVal[eValMap[i]][1]-vVal[eValMap[i]][0]); 
      }
   }

   } // NA > 0
}

void Circuit::SolveMatrix()
{
  
   if (cNum > 0) { // Only if capacitors is B added
   if (!solveOnly) {
      // A + alpha*B
      for (int i=0; i<NA; i++) {
      for (int j=0; j<NA; j++) {
         A(i, j) += 1.0/dt*B(i, j);
      }
      }
      
      // RHS = r - alpha*B*x_{i-1}
      for (int i=0; i<NA; i++) {
      for (int j=0; j<NA; j++) {
         r(i) += 1.0/dt*B(i, j)*x(j);
      }
      }
   } 
   }

   // Gaussian Elimination
   if (NA > 0) {
      gaussian_elimination(A, r);
      back_substitution(A, r, x);
   }
}

void Circuit::AddElement(std::string eTypein, int &e)
{
   // Increment elements
   eNum++;

   // Add to element number map
   eNumMap.push_back(e);

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
      i2 = i / eNum;
      node1 = enMap[i1][i2];
      iflg = 0;

      // Also compute min node at same time
      if (node1 < nNumMin) nNumMin = node1;
   for (int j=i+1; j<eNum2; j++){
      j1 = j % eNum;
      j2 = j / eNum;
      node2 = enMap[j1][j2];
      if (node1 == node2) {
         iflg = 1;
         break;
      }
   }
   if (iflg == 0) nNum++;
   }

   // Since voltage sources/capacitors may have initial voltage, we check if any
   // voltages are at ground AND they are not specified in gNodes
   //
   double rthresh = 1E-6;
   for (int i=0;i<eNum;i++){ // check each element,

      // First, check capacitors
      if (eType[i] == ETYPE_CAPACITOR) { // is it a capacitor?
         for (int j=0;j<2;j++) { // check each capacitor node
            if (abs(v0Val[eValMap[i]][j]) <= rthresh) { // is this node a ground?
               iflg = 0;
               int gNumCurrent = gNodes.size();
               for (int k=0;k<gNumCurrent;k++) {            // have we set it as a ground yet?
                  if (gNodes[k] == enMap[i][j]) iflg = 1; // yes we did
               }
               if (iflg == 0) gNodes.push_back(enMap[i][j]); // no we didn't, so add it
            }
         }
      }

      // Then, check voltage sources
      if (eType[i] == ETYPE_VSOURCE) { // is it a voltage source?
         for (int j=0;j<2;j++) { // check each node
            if (abs(vVal[eValMap[i]][j]) <= rthresh) { // is this node a ground?
               iflg = 0;
               int gNumCurrent = gNodes.size();
               for (int k=0;k<gNumCurrent;k++) {            // have we set it as a ground yet?
                  if (gNodes[k] == enMap[i][j]) iflg = 1; // yes we did
               }
               if (iflg == 0) gNodes.push_back(enMap[i][j]); // no we didn't, so add it
            }
         }
      }
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

void Circuit::AddVoltage(std::vector<double> &data)
{
   vVal.push_back(data);
}

void Circuit::SayInfo()
{
   std::cout << std::endl;
   std::cout << "Circuit summary:"    << std::endl;
   std::cout << "...nodes        : "  << GetNumNodes()      << std::endl;
   std::cout << "   ...grounds   : "  << GetNumGrounds()    << std::endl;
   std::cout << "...elements     : "  << GetNumElements()   << std::endl;
   std::cout << "   ...resistors : "  << GetNumResistors()  << std::endl;
   std::cout << "   ...capacitors: "  << GetNumCapacitors() << std::endl;
   std::cout << "   ...vsources  : "  << GetNumVSources()   << std::endl;
   std::cout << std::endl;
}

void Circuit::SayVoltages()
{

   std::cout << "Circuit voltages:" << std::endl;

   double voltage;
   for (int i=nNumMin; i<nNum+nNumMin; i++){
      voltage = GetNodeVoltage(i);
      std::cout << "...node " << i << ": " << voltage << " V" << std::endl;
   }

   int node1,node2;
   for (int i=0; i<eNum; i++){
      node1 = enMap[i][0]; 
      node2 = enMap[i][1]; 
      voltage = GetNodeVoltage(node2) - GetNodeVoltage(node1);

      if (eType[i] == ETYPE_RESISTOR) {
         std::cout << "..." << 1.0/rVal[eValMap[i]] << " ohm resistor (nodes "     << node1 << " and " << node2 << "): " << voltage << " V" << std::endl;
      } else if (eType[i] == ETYPE_CAPACITOR) {
         std::cout << "..." << cVal[eValMap[i]]     << " farad capacitor (nodes "  << node1 << " and " << node2 << "): " << voltage << " V" << std::endl;
      } else if (eType[i] == ETYPE_VSOURCE) {
         std::cout << "..." << vVal[eValMap[i]][1] - vVal[eValMap[i]][0] << " V voltage source (nodes " << node1 << " and " << node2 << "): " << voltage << " V" << std::endl;
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
      for (int j=0; j<NA; j++) std::cout << A(i, j) << " ";
      std::cout << std::endl;
   }
   std::cout << std::endl;

   std::cout << "B: " << std::endl;
   for (int i=0; i<NA; i++){
      for (int j=0; j<NA; j++) std::cout << B(i, j) << " ";
      std::cout << std::endl;
   }
   std::cout << std::endl;

   std::cout << "r: " << std::endl;
   for (int i=0; i<NA; i++){
      std::cout << r(i) << std::endl;
   }
   std::cout << std::endl;

   std::cout << "x: " << std::endl;
   for (int i=0; i<NA; i++){
      std::cout << x(i) << std::endl;
   }
   std::cout << std::endl;

   std::cout << std::endl;
}

}  // namespace lgr
