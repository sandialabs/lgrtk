#include <lgr_circuit.hpp>
#include <algorithm>
#include <iostream>
#include <fstream>

namespace lgr {

Circuit::Circuit()
{
   nNum    = 0;
   gNum    = 0;
   gNumMax = 0;
   nNumMin = 0;

   eNum    = 0;
   rNum    = 0;
   cNum    = 0;
   lNum    = 0;
   vNum    = 0;
   eMesh   = -1; // < 0 => not yet set

   dt   = 0.0;
   time = 0.0;

   firstCall    = true;
   usingCircuit = false;
   usingMesh    = false;

   nmat_size = 0;
}

Circuit::~Circuit()
{
}

void Circuit::AddResistorUser(int &e, std::vector<int> &nodes, double &con) {
   AddElement("resistor",e);
   AddNodes(nodes);
   AddConductance(con);
}

void Circuit::AddCapacitorUser(int &e, std::vector<int> &nodes, double &cap) {
   AddElement("capacitor",e);
   AddNodes(nodes);
   AddCapacitance(cap);
}

void Circuit::AddInductorUser(int &e, std::vector<int> &nodes, double &ind) {
   AddElement("inductor",e);
   AddNodes(nodes);
   AddInductance(ind);
}

void Circuit::AddFixedVUser(std::vector<int> &nodes, std::vector<double> &vals) {
   fvNodes = nodes;
   fvVals  = vals;
}

void Circuit::AddInitialVUser(std::vector<int> &nodes, std::vector<double> &vals) {
   ivNodes = nodes;
   ivVals  = vals;
}

void Circuit::AddFixedIUser(std::vector<int> &nodes, std::vector<double> &vals) {
   fiNodes = nodes;
   fiVals  = vals;
}

void Circuit::AddInitialIUser(std::vector<int> &nodes, std::vector<double> &vals) {
   iiNodes = nodes;
   iiVals  = vals;
}

void Circuit::AddMeshUser(int &e) {
   usingMesh = true;
   eMesh = e;
}

void Circuit::ParseYAML(Omega_h::InputMap& pl, Omega_h::ExprEnv& env_in)
{

   if (pl.is_map("circuit")) {
      // Alright, we do have a circuit in the yaml file
      auto& circuit_pl = pl.get_map("circuit");

      // Fixed
      if (circuit_pl.is_map("fixed")) {
         auto& fixedv_pl = circuit_pl.get_map("fixed");

         // Add voltage nodes
         if (fixedv_pl.is_list("voltage nodes")){
            auto& fvn_pl = fixedv_pl.get_list("voltage nodes");
            for (int j=0; j < fvn_pl.size(); j++) {
               fvNodes.push_back(fvn_pl.get<int>(j));
            }
         } 

         // Add voltage values
         if (fixedv_pl.is_list("voltage values")){
            auto& fvn_pl = fixedv_pl.get_list("voltage values");
            for (int j=0; j < fvn_pl.size(); j++) {
               auto read_value = fvn_pl.get<std::string>(j);
               auto value = get_double(env_in, read_value);
               fvVals.push_back(value);
            }
         } 

         // Add current nodes
         if (fixedv_pl.is_list("current nodes")){
            auto& fvn_pl = fixedv_pl.get_list("current nodes");
            for (int j=0; j < fvn_pl.size(); j++) {
               fiNodes.push_back(fvn_pl.get<int>(j));
            }
         } 

         // Add current values
         if (fixedv_pl.is_list("current values")){
            auto& fvn_pl = fixedv_pl.get_list("current values");
            for (int j=0; j < fvn_pl.size(); j++) {
               auto read_value = fvn_pl.get<std::string>(j);
               auto value = get_double(env_in, read_value);
               fiVals.push_back(value);
            }
         } 
      }

      // Initial
      if (circuit_pl.is_map("initial")) {
         auto& fixedv_pl = circuit_pl.get_map("initial");

         // Add voltage nodes
         if (fixedv_pl.is_list("voltage nodes")){
            auto& fvn_pl = fixedv_pl.get_list("voltage nodes");
            for (int j=0; j < fvn_pl.size(); j++) {
               ivNodes.push_back(fvn_pl.get<int>(j));
            }
         } 

         // Add voltage values
         if (fixedv_pl.is_list("voltage values")){
            auto& fvn_pl = fixedv_pl.get_list("voltage values");
            for (int j=0; j < fvn_pl.size(); j++) {
               auto read_value = fvn_pl.get<std::string>(j);
               auto value = get_double(env_in, read_value);
               ivVals.push_back(value);
            }
         } 

         // Add current nodes
         if (fixedv_pl.is_list("current nodes")){
            auto& fvn_pl = fixedv_pl.get_list("current nodes");
            for (int j=0; j < fvn_pl.size(); j++) {
               iiNodes.push_back(fvn_pl.get<int>(j));
            }
         } 

         // Add current values
         if (fixedv_pl.is_list("current values")){
            auto& fvn_pl = fixedv_pl.get_list("current values");
            for (int j=0; j < fvn_pl.size(); j++) {
               auto read_value = fvn_pl.get<std::string>(j);
               auto value = get_double(env_in, read_value);
               iiVals.push_back(value);
            }
         } 
      }

      // Mesh element number
      if (circuit_pl.is<int>("mesh element")) {
         int e = circuit_pl.get<int>("mesh element");
         AddMeshUser(e);
      } // Mesh element number

      // Branches
      if (circuit_pl.is_list("branch")) {
         auto& branch_pl = circuit_pl.get_list("branch");

         for (int i=0; i < branch_pl.size(); i++) {
            if (branch_pl.is_map(i)) {
                auto& bmap_pl = branch_pl.get_map(i);

                // Add number of branches
                if (bmap_pl.is<std::string>("count")) {
                   auto read_value = bmap_pl.get<std::string>("count");
                   auto n = get_int(env_in, read_value);
                   Branch b;
                   b.count = n;
                   branch.push_back(b);
                }

                // Add repeated elements numbers in branch
                if (bmap_pl.is_list("elements")){
                   auto& belem_pl = bmap_pl.get_list("elements");
                   auto this_branch = branch.size() - 1;
                   for (int j=0; j < belem_pl.size(); j++) {
                      auto elem = belem_pl.get<int>(j);
                      branch.at(this_branch).elements.push_back(elem);
                   }
                } 
            }
         }
      } // Branches


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
                if (rmap_pl.is<std::string>("conductance")) {
                   auto read_value = rmap_pl.get<std::string>("conductance");
                   auto rcval = get_double(env_in, read_value);
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
                if (cmap_pl.is<std::string>("capacitance")) {
                   auto read_value = cmap_pl.get<std::string>("capacitance");
                   auto ccval = get_double(env_in, read_value);
                   AddCapacitance(ccval);
                }
            }
         }
      } // capacitors

      // Inductors
      if (circuit_pl.is_list("inductors")) {
         auto& inductors_pl = circuit_pl.get_list("inductors");

         for (int i=0; i < inductors_pl.size(); i++) {
            if (inductors_pl.is_map(i)) {
                auto& cmap_pl = inductors_pl.get_map(i);

                // Add element number
                if (cmap_pl.is<int>("element")) {
                   int e = cmap_pl.get<int>("element");
                   AddElement("inductor",e);
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

                // Add inductor inductance
                if (cmap_pl.is<double>("inductance")) {
                   double ccval = cmap_pl.get<double>("inductance");
                   AddInductance(ccval);
                } 
                if (cmap_pl.is<int>("inductance")) {
                   double ccval = static_cast<double>(cmap_pl.get<int>("inductance"));
                   AddInductance(ccval);
                }
                if (cmap_pl.is<std::string>("inductance")) {
                   auto read_value = cmap_pl.get<std::string>("inductance");
                   auto ccval = get_double(env_in, read_value);
                   AddInductance(ccval);
                }
            }
         }
      } // Inductors
   } // Circuit
}

double Circuit::GetMeshAnodeVoltage()
{
   if (!usingMesh) return 1.0;

   int efind = -1;
   try {
      for (int i=0;i<eNum;i++) {
         if (eNumMap[std::size_t(i)] == eMesh) {
            efind = i;
            break;
         }
      }
      if (efind < 0) {
         throw 1; // Invalid mesh element number
      } else if (eType[std::size_t(efind)] != ETYPE_RESISTOR) {
         throw 2; // Valid element number, but e is not resistor and thus not a mesh
      } 
   }
   catch (int ex) {
      std::cout << "Circuit solve GetMeshAnodeVoltage() execption " << ex << std::endl;
   }

   return GetNodeVoltage(enMap[std::size_t(efind)][0]);
}

double Circuit::GetMeshCathodeVoltage()
{
   if (!usingMesh) return 0.0;

   int efind = -1;
   try {
      for (int i=0;i<eNum;i++) {
         if (eNumMap[std::size_t(i)] == eMesh) {
            efind = i;
            break;
         }
      }
      if (efind < 0) {
         throw 1; // Invalid mesh element number
      } else if (eType[std::size_t(efind)] != ETYPE_RESISTOR) {
         throw 2; // Valid element number, but e is not resistor and thus not a mesh
      }
   }
   catch (int ex) {
      std::cout << "Circuit solve GetMeshCathodVoltage() execption " << ex << std::endl;
   }

   return GetNodeVoltage(enMap[std::size_t(efind)][1]);
}

double Circuit::GetMeshVoltageDrop()
{
    return GetMeshAnodeVoltage() - GetMeshCathodeVoltage();
}

double Circuit::GetMeshConductance()
{
   if (!usingMesh) return 0.0;

   double conductance = GetElementConductance(eMesh);

   for (auto it = branch.begin(); it != branch.end(); ++it) {
      for (auto eit = (*it).elements.begin(); eit != (*it).elements.end(); ++eit) {
         if ( (*eit) == eMesh) {
            // This is a single branches mesh conductance, not the combined branch
            conductance *= 1.0/(*it).count;
         }
      }
    }

   return conductance;
}

double Circuit::GetMeshCurrent()
{
   if (!usingMesh) return 0.0;

   // This is a single branches current, since GetMeshConductance scales by the count
   return GetMeshVoltageDrop()*GetMeshConductance();
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
      if (nodein == gNodes[std::size_t(i)]) {
         iflg = 1;
         break;
      }
   }

   double voltage;
   if (iflg == 1) {
      voltage = 0.0; 
   } else {
      int ieqn = ConvertNodeToEq<int>(nodein);
      voltage = x_vector(ieqn);
   }

   return voltage;
}

void Circuit::SetElementConductance(int e, double c)
{
   int efind = -1;
   try {
      for (int i=0;i<eNum;i++) {
         if (eNumMap[std::size_t(i)] == e) {
            efind = i;
            break;
         }
      }
      if (efind < 0) {
         throw 1; // Invalid element number
      } else if (eType[std::size_t(efind)] != ETYPE_RESISTOR) {
         throw 2; // Valid element number, but e is not resistor
      } else if (c <= 0) {
         throw 3; // Invalid conductance
      }
   }
   catch (int ex) {
      std::cout << "Circuit solve SetElementConductance() execption " << ex << std::endl;
   }

   rVal[std::size_t(eValMap[std::size_t(efind)])] = c;
}

double Circuit::GetElementConductance(int e)
{
   int efind = -1;
   try {
      for (int i=0;i<eNum;i++) {
         if (eNumMap[std::size_t(i)] == e) {
            efind = i;
            break;
         }
      }
      if (efind < 0) {
         throw 1; // Invalid element number
      } else if (eType[std::size_t(efind)] != ETYPE_RESISTOR) {
         throw 2; // Valid element number, but e is not resistor
      }
   }
   catch (int ex) {
      std::cout << "Circuit solve GetElementConductance() execption " << ex << std::endl;
   }

   return rVal[std::size_t(eValMap[std::size_t(efind)])];
}

void Circuit::SetMeshConductance(double c)
{
   if (usingMesh) {

   double conductance = c;

   for (auto it = branch.begin(); it != branch.end(); ++it) {
      for (auto eit = (*it).elements.begin(); eit != (*it).elements.end(); ++eit) {
         // Note this does not use eNumMap since SetElementConductance will do that below;
         // This is pure user driven
         if ( (*eit) == eMesh) {
            conductance *= (*it).count;
         }
      }
    }

   SetElementConductance(eMesh, conductance);
   }
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

int Circuit::GetNumInductors()
{
   return lNum;
}

int Circuit::GetNumGrounds()
{
   return gNum;
}

void Circuit::Setup(Omega_h::ExprEnv& env_in, Omega_h::InputMap& pl)
{
   ParseYAML(pl, env_in);
   Setup();
}

double Circuit::get_double(Omega_h::ExprEnv& env_in, std::string& expr) {
    Omega_h::ExprOpsReader reader;
    auto op = reader.read_ops(expr);
    auto const value_any = op->eval(env_in);
    return Omega_h::any_cast<double>(value_any);
}

int Circuit::get_int(Omega_h::ExprEnv& env_in, std::string& expr) {
  return static_cast<int>(get_double(env_in, expr));
}

void Circuit::Setup()
{
   if (eNum > 0) {
      ComponentMap();
      NodeCount();
   }
   UpdateGrounds();
   UpdateMatrixSize();
   ZeroMatrix();
   SayInfo();

   firstCall = false;

   UpdateGrounds();
   UpdateMatrixSize();
   UpdateBranchValues();

   InitOutputValues();

}

void Circuit::InitOutputValues() {
   for (int i=0; i<eNum; i++){
      // eval is user element number mapping
      int eval = eNumMap[std::size_t(i)];
      int node1 = enMap[std::size_t(i)][0];
      int node2 = enMap[std::size_t(i)][1];
      double voltage = GetNodeVoltage(node2) - GetNodeVoltage(node1);

      // Element voltages
      element_voltages.insert(std::pair<int, double>(eval, voltage));

      // Currents (assume initially zero)
      element_currents.insert(std::pair<int, double>(i, 0.0));
   }
}

void Circuit::UpdateOutputValues() {
   for (int i=0; i<eNum; i++){
      // eval is user element number mapping
      int eval = eNumMap[std::size_t(i)];
      int node1 = enMap[std::size_t(i)][0];
      int node2 = enMap[std::size_t(i)][1];

      // Voltages
      auto it = element_voltages.find(eval);
      if (it != element_voltages.end()) {
         double voltage = GetNodeVoltage(node2) - GetNodeVoltage(node1);
         it->second = voltage;
      }

      // Currents
      it = element_currents.find(eval);
      if (it != element_currents.end()) {
         std::size_t ecomp = static_cast<std::size_t>(eValMap[std::size_t(i)]);

         // Resistor:
         //   I = G*V
         if (eType[std::size_t(i)] == ETYPE_RESISTOR) {
            double voltage = GetNodeVoltage(node1) - GetNodeVoltage(node2);
            it->second = rVal[ecomp]*voltage;

         // Capacitor:
         //   I = C*dV/dt
         } else if (eType[std::size_t(i)] == ETYPE_CAPACITOR) {
            int ieqn1 = ConvertNodeToEq<int>(node1);
            int ieqn2 = ConvertNodeToEq<int>(node2);
            double this_voltage = x_vector(ieqn1) - x_vector(ieqn2);
            double last_voltage = x_vector_last(ieqn1) - x_vector_last(ieqn2);
            double dVdt = (this_voltage - last_voltage)/dt;
            it->second = cVal[ecomp]*dVdt;

         // Inductor:
         //   I = from solution vector
         } else if (eType[std::size_t(i)] == ETYPE_INDUCTOR) {
            // Find which inductor this is
            int il = 0;
            for (int j=0;j<eNum;j++) {
               if (eNumMap[std::size_t(j)] == it->first) break;
               if (eType[std::size_t(j)] == ETYPE_INDUCTOR) il += 1;
            }
            int jshift = nNum-gNum;
            it->second = x_vector(jshift+il);
         }
      }
   }
}

void Circuit::Solve(double dtin, double timein)
{

   if (usingCircuit) {
      dt   = dtin;
      time = timein;

      AssembleMatrix();
      if ( (dt > 0) || (cNum + lNum == 0) ) {
         SolveMatrix();
         UpdateOutputValues();
      }
   }
}

void Circuit::UpdateBranchValues()
{
    for (auto it = branch.begin(); it != branch.end(); ++it) {
       for (auto eit = (*it).elements.begin(); eit != (*it).elements.end(); ++eit) {

          int e = (*eit);
          int efind = -1;
          for (int i=0;i<eNum;i++) {
             if (eNumMap[std::size_t(i)] == e) {
                efind = i;
                break;
             }
          }
          if (efind == -1) {
             printf("Circuit could not find element %d\n",e);
             OMEGA_H_CHECK(0 > 1);
          }

          std::size_t ecomp = static_cast<std::size_t>(eValMap[std::size_t(efind)]);

          if (eType[std::size_t(efind)] == ETYPE_RESISTOR) {
             // Modify conductance (inverse resistance)
             rVal[ecomp] *= (*it).count;
          } else if (eType[std::size_t(efind)] == ETYPE_CAPACITOR) {
             // Modify capacitance
             cVal[ecomp] *= (*it).count;
          } else if (eType[std::size_t(efind)] == ETYPE_INDUCTOR) {
             // Modify inductance
             lVal[ecomp] *= 1.0/(*it).count;
          }
       }

    }
}

void Circuit::UpdateMatrixSize()
{
   // Count number of voltage constraints
   int vnd = 0;
   for (std::size_t i = 0; i<fvNodes.size(); i++) {
      if (abs(fvVals[i]) > rthresh) vnd++;
   }
   if (firstCall) {
      for (std::size_t i = 0; i<ivNodes.size(); i++) {
         if (abs(ivVals[i]) > rthresh) vnd++;
      }
   }

   // Total matrix size
   nmat_size = nNum - gNum + lNum + vnd;

   if (nmat_size > 0) {
      usingCircuit = true;
   } else {
      usingCircuit = false;
   }

   if (usingCircuit) {

      // Setup the array memory
      if (firstCall) {
         N_matrix = MediumMatrix(nmat_size);
         M_matrix = MediumMatrix(nmat_size);
         NM_matrix = MediumMatrix(nmat_size);
         b_vector = MediumVector(nmat_size);
         x_vector = MediumVector(nmat_size);
         x_vector_last = MediumVector(nmat_size);
      } else {
         N_matrix.size = nmat_size;
         M_matrix.size = nmat_size;
         NM_matrix.size = nmat_size;
         N_matrix.entries.resize(std::size_t(nmat_size*nmat_size));
         M_matrix.entries.resize(std::size_t(nmat_size*nmat_size));
         NM_matrix.entries.resize(std::size_t(nmat_size*nmat_size));
      }
      
      AR_matrix.resize(std::size_t(nNum-gNum), std::vector<int>(std::size_t(rNum)));
      G_matrix.resize (std::size_t(rNum), std::vector<double>(std::size_t(rNum)));
      AC_matrix.resize(std::size_t(nNum-gNum), std::vector<int>(std::size_t(cNum)));
      C_matrix.resize (std::size_t(cNum), std::vector<double>(std::size_t(cNum)));
      AL_matrix.resize(std::size_t(nNum-gNum), std::vector<int>(std::size_t(lNum)));
      L_matrix.resize (std::size_t(lNum), std::vector<double>(std::size_t(lNum)));
      AV_matrix.resize(std::size_t(nNum-gNum), std::vector<int>(std::size_t(vNum)));
      V_vector.resize (std::size_t(vNum));
      I_vector.resize (std::size_t(nNum-gNum));
   }
}

void Circuit::ZeroMatrix() {

   for (int i=0; i< nmat_size; i++) {
   for (int j=0; j< nmat_size; j++) {
      N_matrix(i,j) = 0.0;
      M_matrix(i,j) = 0.0;
      NM_matrix(i,j) = 0.0;
   }
   }
   for (int i=0; i<nmat_size; i++) {
      b_vector(i) = 0.0;
   }
   // Initial condition and only on first call
   if (firstCall) {
      for (int i = 0; i<nmat_size; i++) {
         x_vector(i) = 0.0;
         x_vector_last(i) = 0.0;
      }
   
      for (std::size_t i = 0; i<ivNodes.size(); i++) {
         int ieqn = ConvertNodeToEq<int>(ivNodes[i]);
         if (abs(ivVals[i]) > rthresh) x_vector(ieqn) = ivVals[i];
      }
      for (std::size_t i = 0; i<fvNodes.size(); i++) {
         int ieqn = ConvertNodeToEq<int>(fvNodes[i]);
         if (abs(fvVals[i]) > rthresh) x_vector(ieqn) = fvVals[i];
      }
      for (std::size_t i = 0; i<iiNodes.size(); i++) {
         int ieqn = ConvertNodeToEq<int>(iiNodes[i]);
         ieqn+=nNum-gNum; // shift
         x_vector(ieqn) = iiVals[i];
      }
      for (std::size_t i = 0; i<fiNodes.size(); i++) {
         int ieqn = ConvertNodeToEq<int>(fiNodes[i]);
         ieqn+=nNum-gNum; // shift
         x_vector(ieqn) = fiVals[i];
      }
   }

   for (std::size_t i=0; i<std::size_t(nNum-gNum); i++) {
   for (std::size_t j=0; j<std::size_t(rNum); j++) {
      AR_matrix[i][j] = 0;
   }
   }
   for (std::size_t i=0; i<std::size_t(rNum); i++) {
   for (std::size_t j=0; j<std::size_t(rNum); j++) {
      G_matrix[i][j] = 0.0;
   }
   }

   for (std::size_t i=0; i<std::size_t(nNum-gNum); i++) {
   for (std::size_t j=0; j<std::size_t(cNum); j++) {
      AC_matrix[i][j] = 0;
   }
   }
   for (std::size_t i=0; i<std::size_t(cNum); i++) {
   for (std::size_t j=0; j<std::size_t(cNum); j++) {
      C_matrix[i][j] = 0.0;
   }
   }

   for (std::size_t i=0; i<std::size_t(nNum-gNum); i++) {
   for (std::size_t j=0; j<std::size_t(lNum); j++) {
      AL_matrix[i][j] = 0;
   }
   }
   for (std::size_t i=0; i<std::size_t(lNum); i++) {
   for (std::size_t j=0; j<std::size_t(lNum); j++) {
      L_matrix[i][j] = 0.0;
   }
   }

   for (std::size_t i=0; i<std::size_t(nNum-gNum); i++) {
   for (std::size_t j=0; j<std::size_t(vNum); j++) {
      AV_matrix[i][j] = 0;
   }
   }
   for (std::size_t i=0; i<std::size_t(vNum); i++) {
      V_vector[i] = 0.0;
   }

   for (std::size_t i=0; i<std::size_t(nNum-gNum); i++) {
      I_vector[i] = 0.0;
   }
}

void Circuit::AssembleRMatrix() {

   int ir=-1;
   // Loop over elements
   for (std::size_t i_t=0; i_t<std::size_t(eNum); i_t++) {

   std::size_t ibr_t;
   auto eVM_t = std::size_t(eValMap[i_t]);

   if (eType[i_t] == ETYPE_RESISTOR) {
      ir++;
      ibr_t = std::size_t(ir);
      G_matrix[ibr_t][ibr_t] = rVal[eVM_t];

   for (std::size_t j_t=0; j_t<2   ; j_t++) {

      int iflg = 0;
      int node = enMap[i_t][j_t];

      for (std::size_t k_t=0; k_t<std::size_t(gNum); k_t++) {
         if (node == gNodes[k_t]) {
            iflg = 1;
            break;
         }
      }

      if (iflg == 0) {

         int sign = 1; // Start node
         if (j_t == 1) sign = -1; // End node
         
         std::size_t ieqn = ConvertNodeToEq<std::size_t>(node);
         AR_matrix[ieqn][ibr_t] = sign;
      }
   }
   }
   }
}

void Circuit::AssembleCMatrix() {

   int ic=-1;
   // Loop over elements
   for (std::size_t i_t=0; i_t<std::size_t(eNum); i_t++) {

   std::size_t ibr_t;
   auto eVM_t = std::size_t(eValMap[i_t]);

   if (eType[i_t] == ETYPE_CAPACITOR) {
      ic++;
      ibr_t = std::size_t(ic);
      C_matrix[ibr_t][ibr_t] = cVal[eVM_t];

   for (std::size_t j_t=0; j_t<2   ; j_t++) {

      int iflg = 0;
      int node = enMap[i_t][j_t];

      for (std::size_t k_t=0; k_t<std::size_t(gNum); k_t++) {
         if (node == gNodes[k_t]) {
            iflg = 1;
            break;
         }
      }

      if (iflg == 0) {

         int sign = 1; // Start node
         if (j_t == 1) sign = -1; // End node
         
         std::size_t ieqn = ConvertNodeToEq<std::size_t>(node);
         AC_matrix[ieqn][ibr_t] = sign;
      }
   }
   }
   }
}

void Circuit::AssembleLMatrix() {

   int il=-1;
   // Loop over elements
   for (std::size_t i_t=0; i_t<std::size_t(eNum); i_t++) {

   std::size_t ibr_t;
   auto eVM_t = std::size_t(eValMap[i_t]);

   if (eType[i_t] == ETYPE_INDUCTOR) {
      il++;
      ibr_t = std::size_t(il);
      L_matrix[ibr_t][ibr_t] = lVal[eVM_t];

   for (std::size_t j_t=0; j_t<2   ; j_t++) {

      int iflg = 0;
      int node = enMap[i_t][j_t];

      for (std::size_t k_t=0; k_t<std::size_t(gNum); k_t++) {
         if (node == gNodes[k_t]) {
            iflg = 1;
            break;
         }
      }

      if (iflg == 0) {

         int sign = 1; // Start node
         if (j_t == 1) sign = -1; // End node
         
         std::size_t ieqn = ConvertNodeToEq<std::size_t>(node);
         AL_matrix[ieqn][ibr_t] = sign;
      }
   }

   }
   }
}

void Circuit::AssembleSourceMatrix()
{
   // fixed voltage sources
   int iv=-1;
   for (std::size_t i=0;i< fvVals.size(); i++) {
      if (abs(fvVals[i]) >= rthresh) {
         iv++;
         auto nn_t = std::size_t(iv);
         V_vector[nn_t] = fvVals[i];
         std::size_t ieqn = ConvertNodeToEq<std::size_t>(fvNodes[i]);
         AV_matrix[ieqn][nn_t] = 1;
      }
   }

   // initial voltage sources
   if (firstCall) {
   for (std::size_t i=0;i< ivVals.size(); i++) {
      if (abs(ivVals[i]) >= rthresh) {
         iv++;
         auto nn_t = std::size_t(iv);
         V_vector[nn_t] = ivVals[i];
         std::size_t ieqn = ConvertNodeToEq<std::size_t>(ivNodes[i]);
         AV_matrix[ieqn][nn_t] = 1;
      }
   }
   }

   // Should add search below to not add if ground node check
   // fixed current sources
   for (std::size_t i=0; i<fiVals.size(); i++) {
      std::size_t ieqn = ConvertNodeToEq<std::size_t>(fiNodes[i]);
      I_vector[ieqn] = fiVals[i];
   }

   // initial current sources
   if (firstCall) {
   for (std::size_t i=0; i<iiVals.size(); i++) {
      std::size_t ieqn = ConvertNodeToEq<std::size_t>(iiNodes[i]);
      I_vector[ieqn] = iiVals[i];
   }
   }
}

void Circuit::AssembleNMatrix()
{

   // Put in AR*R*AR'
   std::vector< std::vector<double> > GA(std::size_t(rNum), std::vector<double>(std::size_t(nNum-gNum)));

   int ishift = 0;
   int jshift = 0;
   for (int i=0;i<rNum     ;i++) {
   for (int j=0;j<nNum-gNum;j++) {
      for (int k=0; k<rNum; k++) {
         GA[std::size_t(i)][std::size_t(j)]+= G_matrix[std::size_t(i)][std::size_t(k)]*AR_matrix[std::size_t(j)][std::size_t(k)];
      }
   }
   }
   for (int i=0;i<nNum-gNum;i++) {
   for (int j=0;j<nNum-gNum;j++) {
      N_matrix(ishift+i,jshift+j) = 0.0;
      for (int k=0; k<rNum; k++) {
         N_matrix(ishift+i,jshift+j)+=AR_matrix[std::size_t(i)][std::size_t(k)]*GA[std::size_t(k)][std::size_t(j)];
      }
   }
   }

   // Put in AL
   ishift = 0;
   jshift = nNum-gNum;
   for (int i=0;i<nNum-gNum;i++) {
   for (int j=0;j<lNum;j++) {
      N_matrix(ishift+i,jshift+j) = AL_matrix[std::size_t(i)][std::size_t(j)];
   }
   }

   // Put in AL'
   ishift = nNum-gNum;
   jshift = 0;
   for (int i=0;i<nNum-gNum;i++) {
   for (int j=0;j<lNum;j++) {
      N_matrix(ishift+j,jshift+i) = -dt*AL_matrix[std::size_t(i)][std::size_t(j)];
   }
   }

   // Put in AV
   ishift = 0;
   jshift = nNum-gNum+lNum;
   for (int i=0;i<nNum-gNum;i++) {
   for (int j=0;j<vNum;j++) {
      N_matrix(ishift+i,jshift+j) = AV_matrix[std::size_t(i)][std::size_t(j)];
   }
   }

   // Put in AV'
   ishift = nNum-gNum+lNum;
   jshift = 0;
   for (int i=0;i<nNum-gNum;i++) {
   for (int j=0;j<vNum;j++) {
      N_matrix(ishift+j,jshift+i) = AV_matrix[std::size_t(i)][std::size_t(j)];
   }
   }

   ModifyCEquation();
}

void Circuit::ModifyCEquation()
{

   std::vector<int> cnused;
   for (std::size_t i_t=0; i_t<std::size_t(eNum); i_t++){
   if (eType[i_t] == ETYPE_CAPACITOR) {
   for (std::size_t j_t=0; j_t<2   ; j_t++){
      int iflg = 0;

      int node = enMap[i_t][j_t];

      for (std::size_t k_t=0; k_t<std::size_t(gNum); k_t++) {
         if (node == gNodes[k_t]) {
            iflg = 1;
            break;
         }
      }

      if (iflg == 0) {
         for (std::size_t k_t=0; k_t < cnused.size(); k_t++) {
            if(node == cnused[k_t]) {
               iflg = 1;
               break;
            }
         }
      }

      if (iflg == 0) {
         cnused.push_back(node);

         int ieqn = ConvertNodeToEq<int>(node);
         
         // dt formalism
         for (int k=0; k < nmat_size; k++) {
            N_matrix(ieqn, k) = N_matrix(ieqn, k)*dt;
         }
         I_vector[std::size_t(ieqn)] = I_vector[std::size_t(ieqn)]*dt;
      }
   }
   }
   }
}

void Circuit::AssembleMMatrix()
{

   std::vector< std::vector<double> > CA(std::size_t(cNum), std::vector<double>(std::size_t(nNum-gNum)));

   int ishift = 0;
   int jshift = 0;
   for (int i=0;i<cNum     ;i++) {
   for (int j=0;j<nNum-gNum;j++) {
      for (int k=0; k<cNum; k++) {
         CA[std::size_t(i)][std::size_t(j)]+= C_matrix[std::size_t(i)][std::size_t(k)]*AC_matrix[std::size_t(j)][std::size_t(k)];
      }
   }
   }
   for (int i=0;i<nNum-gNum;i++) {
   for (int j=0;j<nNum-gNum;j++) {
      M_matrix(ishift+i,jshift+j) = 0.0;
      for (int k=0; k<cNum; k++) {
         M_matrix(ishift+i,jshift+j)+=AC_matrix[std::size_t(i)][std::size_t(k)]*CA[std::size_t(k)][std::size_t(j)];
      }
   }
   }
   ishift = nNum-gNum;
   jshift = nNum-gNum;
   for (int i=0;i<lNum;i++) {
   for (int j=0;j<lNum;j++) {
      M_matrix(ishift+i,jshift+j) = L_matrix[std::size_t(i)][std::size_t(j)];
   }
   }
}

void Circuit::AssembleNMMatrix()
{
   for (int i=0;i<nmat_size;i++) {
   for (int j=0;j<nmat_size;j++) {
      NM_matrix(i,j) = N_matrix(i,j) + M_matrix(i,j);
   }
   }
}

void Circuit::AssemblebVector()
{
   int ishift = 0;
   for (int i=0;i<nNum-gNum; i++) {
      b_vector(ishift+i) = - I_vector[std::size_t(i)];
   }
   ishift = nNum-gNum+lNum;
   for (int i=0;i<vNum;i++) {
      b_vector(ishift+i) = V_vector[std::size_t(i)];
   }
      
   for (int i=0;i<nmat_size;i++) {
   for (int j=0;j<nmat_size;j++) {
      b_vector(i) += M_matrix(i,j)*x_vector(j);
   }
   }

}


void Circuit::AssembleMatrix()
{
   ZeroMatrix();

   AssembleRMatrix();
   AssembleCMatrix();
   AssembleLMatrix();
   AssembleSourceMatrix();

   AssembleNMatrix();
   AssembleMMatrix();
   AssembleNMMatrix();

   AssemblebVector();
}

void Circuit::SolveMatrix()
{
   for (int j=0;j<nmat_size;j++) {
       x_vector_last(j) = x_vector(j);
   }

   gaussian_elimination(NM_matrix, b_vector);
   back_substitution(NM_matrix, b_vector, x_vector);
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
   } else if (eTypein.compare("capacitor") == 0) {
      cNum++;
      eType.push_back(ETYPE_CAPACITOR);
   } else if (eTypein.compare("inductor") == 0) {
      lNum++;
      eType.push_back(ETYPE_INDUCTOR);
   }
}

template <typename T>
T Circuit::ConvertNodeToEq(int node)
{
   node = node - nNumMin;
   if (node >= gNumMax - nNumMin) node = node - gNum;
   return T(node);
}

void Circuit::ComponentMap()
{
   int ir = -1;
   int ic = -1;
   int il = -1;
   for (int i=0; i<eNum; i++){
      if (eType[std::size_t(i)] == ETYPE_RESISTOR) {
         ir++;
         eValMap.push_back(ir);
      } else if (eType[std::size_t(i)] == ETYPE_CAPACITOR) {
         ic++;
         eValMap.push_back(ic);
      } else if (eType[std::size_t(i)] == ETYPE_INDUCTOR) {
         il++;
         eValMap.push_back(il);
      }
   }
}

void Circuit::UpdateGrounds()
{
   gNodes.clear();
   vNum = 0;

   // ground nodes
   for (std::size_t j=0; j <fvNodes.size(); j++) {
      double nodev = fvVals[j];
      if (abs(nodev) <= rthresh ) {
         gNodes.push_back(fvNodes[j]);
      } else {
         vNum++;
      }
   }

   if (firstCall) {
   for (std::size_t j=0; j <ivNodes.size(); j++) {
      double nodev = ivVals[j];
      if (abs(nodev) <= rthresh ) {
         gNodes.push_back(ivNodes[j]);
      } else {
         vNum++;
      }
   }
   }

   // Count ground nodes
   gNum = int(gNodes.size());
   if (gNum == 0) {
      gNumMax = nNum;
   } else {
      for (int i=0; i<gNum; i++){
         if (gNodes[std::size_t(i)] > gNumMax) gNumMax = gNodes[std::size_t(i)];
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
      node1 = enMap[std::size_t(i1)][std::size_t(i2)];
      iflg = 0;

      // Also compute min node at same time
      if (node1 < nNumMin) nNumMin = node1;
   for (int j=i+1; j<eNum2; j++){
      j1 = j % eNum;
      j2 = j / eNum;
      node2 = enMap[std::size_t(j1)][std::size_t(j2)];
      if (node1 == node2) {
         iflg = 1;
         break;
      }
   }
   if (iflg == 0) nNum++;
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

void Circuit::AddInductance(double &data)
{
   lVal.push_back(data);
}

void Circuit::SayInfo()
{
   if (usingCircuit) {
      std::cout << std::endl;
      std::cout << "Circuit summary:"    << std::endl;
      std::cout << "...nodes        : "  << GetNumNodes()      << std::endl;
      std::cout << "   ...grounds   : "  << GetNumGrounds()    << std::endl;
      std::cout << "...elements     : "  << GetNumElements()   << std::endl;
      std::cout << "   ...resistors : "  << GetNumResistors()  << std::endl;
      std::cout << "   ...capacitors: "  << GetNumCapacitors() << std::endl;
      std::cout << "   ...inductors : "  << GetNumInductors()  << std::endl;
      std::cout << std::endl;
   } else {
      std::cout << "Circuit summary:"            << std::endl;
      std::cout << "   ...not using circuit : "  << std::endl;
   }
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
      node1 = enMap[std::size_t(i)][0]; 
      node2 = enMap[std::size_t(i)][1]; 
      voltage = GetNodeVoltage(node1) - GetNodeVoltage(node2);

      if (eType[std::size_t(i)] == ETYPE_RESISTOR) {
         std::cout << "..." << 1.0/rVal[std::size_t(eValMap[std::size_t(i)])] << " ohm resistor (nodes "     << node1 << " minus " << node2 << "): " << voltage << " V" << std::endl;
      } else if (eType[std::size_t(i)] == ETYPE_CAPACITOR) {
         std::cout << "..." << cVal[std::size_t(eValMap[std::size_t(i)])]     << " farad capacitor (nodes "  << node1 << " minus " << node2 << "): " << voltage << " V" << std::endl;
      } else if (eType[std::size_t(i)] == ETYPE_INDUCTOR) {
         std::cout << "..." << lVal[std::size_t(eValMap[std::size_t(i)])]     << " Henry inductor (nodes "  << node1 << " minus " << node2 << "): " << voltage << " V" << std::endl;
      }
   }

   std::cout << std::endl;
}

void Circuit::SayMatrix()
{

   std::cout << "*************************************" << std::endl;
   std::cout << "**** CIRCUIT MATRIX INFORMATION: ****" << std::endl;
   std::cout << "*************************************" << std::endl;


   std::cout << "N: " << std::endl;
   for (int i=0; i<nmat_size; i++){
      for (int j=0; j<nmat_size; j++) std::cout << N_matrix(i, j) << " ";
      std::cout << std::endl;
   }
   std::cout << std::endl;

   std::cout << "M: " << std::endl;
   for (int i=0; i<nmat_size; i++){
      for (int j=0; j<nmat_size; j++) std::cout << M_matrix(i, j) << " ";
      std::cout << std::endl;
   }
   std::cout << std::endl;

   std::cout << "NM: " << std::endl;
   for (int i=0; i<nmat_size; i++){
      for (int j=0; j<nmat_size; j++) std::cout << NM_matrix(i, j) << " ";
      std::cout << std::endl;
   }
   std::cout << std::endl;

   std::cout << "b: " << std::endl;
   for (int i=0; i<nmat_size; i++) {
      std::cout << b_vector(i) << std::endl;
   }
   std::cout << std::endl;

   std::cout << "x: " << std::endl;
   for (int i=0; i<nmat_size; i++) {
      std::cout << x_vector(i) << std::endl;
   }
   std::cout << std::endl;

   std::cout << "AR: " << std::endl;
   for (int i=0; i<nNum-gNum; i++){
      for (int j=0; j<rNum; j++) std::cout << AR_matrix[std::size_t(i)][std::size_t(j)] << " ";
      std::cout << std::endl;
   }
   std::cout << std::endl;

   std::cout << "G: " << std::endl;
   for (int i=0; i<rNum; i++){
      for (int j=0; j<rNum; j++) std::cout << G_matrix[std::size_t(i)][std::size_t(j)] << " ";
      std::cout << std::endl;
   }
   std::cout << std::endl;

   std::cout << "AC: " << std::endl;
   for (int i=0; i<nNum-gNum; i++){
      for (int j=0; j<cNum; j++) std::cout << AC_matrix[std::size_t(i)][std::size_t(j)] << " ";
      std::cout << std::endl;
   }
   std::cout << std::endl;

   std::cout << "C: " << std::endl;
   for (int i=0; i<cNum; i++){
      for (int j=0; j<cNum; j++) std::cout << C_matrix[std::size_t(i)][std::size_t(j)] << " ";
      std::cout << std::endl;
   }
   std::cout << std::endl;

   std::cout << "AV: " << std::endl;
   for (int i=0; i<nNum-gNum; i++){
      for (int j=0; j<vNum; j++) std::cout << AV_matrix[std::size_t(i)][std::size_t(j)] << " ";
      std::cout << std::endl;
   }
   std::cout << std::endl;

   std::cout << "V: " << std::endl;
   for (int i=0; i<vNum; i++){
      std::cout << V_vector[std::size_t(i)] << std::endl;
   }
   std::cout << std::endl;

   std::cout << "AL: " << std::endl;
   for (int i=0; i<nNum-gNum; i++){
      for (int j=0; j<lNum; j++) std::cout << AL_matrix[std::size_t(i)][std::size_t(j)] << " ";
      std::cout << std::endl;
   }
   std::cout << std::endl;

   std::cout << "L: " << std::endl;
   for (int i=0; i<lNum; i++){
      for (int j=0; j<lNum; j++) std::cout << L_matrix[std::size_t(i)][std::size_t(j)] << " ";
      std::cout << std::endl;
   }
   std::cout << std::endl;


   std::cout << std::endl;
}

}  // namespace lgr
