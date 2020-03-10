#include <lgr_circuit.hpp>
#include <Omega_h_scalar.hpp>
#include "lgr_gtest.hpp"
#include <cmath>
#include <vector>

TEST(circuit, RL_CHARGE) {

/*

YAML input:

circuit:
  mesh element: 0
  fixed:
    voltage nodes: [0,1]
    voltage values: [0.0,3.0]

  resistors:
    - {element: 0, nodes: [2,1], conductance: 0.1}
  inductors:
    - {element: 1, nodes: [0,2], inductance: 10.0}

*/

   std::vector<int>    fixedv_nodes  = {0,1};
   std::vector<double> fixedv_values = {0.0,3.0};

   int e0 = 0;
   std::vector<int> nodes0 = {2,1};
   double conductance = 0.1;

   int e1 = 1;
   std::vector<int> nodes1 = {0,2};
   double inductance = 10.0;

   lgr::Circuit circuit;
   circuit.AddFixedVUser  (fixedv_nodes, fixedv_values);
   circuit.AddResistorUser (e0, nodes0, conductance);
   circuit.AddInductorUser (e1, nodes1, inductance);
   circuit.AddMeshUser     (e0);
   circuit.Setup();

   // Advance to t = 1 sec
   double dt = 0.01;
   double tfinal = 1.0;
   int nstep = int(tfinal/dt); 
   double time = 0.0;
   for (int i=0;i<nstep;i++){
      time+=dt;
      circuit.Solve(dt,time);
   }

   double vna = circuit.GetMeshAnodeVoltage();
   double vnc = circuit.GetMeshCathodeVoltage();
   double vdrop = vnc - vna;

   double vdrop_expect = fixedv_values[1]*(1-exp(-tfinal));

   double thresh = dt;

   EXPECT_TRUE(Omega_h::are_close(vdrop, vdrop_expect, thresh, 0.0));
}

TEST(circuit, ISOURCE) {

/*

YAML input:

circuit:
  mesh element: 0
  fixed:
    voltage nodes: [0]
    voltage values: [0.0]
    current nodes: [1]
    current values: [5.0]
 
  resistors:
    - {element: 0, nodes: [0,1], conductance: 0.1}

*/

   std::vector<int>    fixedv_nodes  = {0};
   std::vector<double> fixedv_values = {0.0};
   std::vector<int>    fixedi_nodes  = {1};
   std::vector<double> fixedi_values = {5.0};

   int e0 = 0;
   std::vector<int> nodes0 = {0,1};
   double conductance = 0.1;

   lgr::Circuit circuit;
   circuit.AddFixedVUser  (fixedv_nodes, fixedv_values);
   circuit.AddFixedIUser  (fixedi_nodes, fixedi_values);
   circuit.AddResistorUser (e0, nodes0, conductance);
   circuit.AddMeshUser     (e0);
   circuit.Setup();

   // Do not advance, but really doesn't matter since no C/L components
   double dt = 0.0;
   circuit.Solve(dt,0.0);

   double vna = circuit.GetMeshAnodeVoltage();
   double vnc = circuit.GetMeshCathodeVoltage();
   double vdrop = vnc - vna;

   double vdrop_expect = -fixedi_values[0]/conductance;

   double thresh = 1E-6;

   EXPECT_TRUE(Omega_h::are_close(vdrop, vdrop_expect, thresh, 0.0));
}

TEST(circuit, RC_DISCHARGE) {

/*

YAML input:

circuit:
  mesh element: 1
  initial:
    voltage nodes: [0]
    voltage values: [3.0]
  fixed:
    voltage nodes: [1]
    voltage values: [0.0]

  capacitors:
    - {element: 0, nodes: [1,0], capacitance: 0.1}

  resistors:
    - {element: 1, nodes: [0,1], conductance: 0.1}

*/

   std::vector<int>    fixedv_nodes  = {1};
   std::vector<double> fixedv_values = {0.0};
   std::vector<int>    initv_nodes   = {0};
   std::vector<double> initv_values  = {3.0};

   int e0 = 0;
   std::vector<int> nodes0 = {1,0};
   double capacitance = 0.1;

   int e1 = 1;
   std::vector<int> nodes1 = {0,1};
   double conductance = 0.1;

   lgr::Circuit circuit;
   circuit.AddFixedVUser  (fixedv_nodes, fixedv_values);
   circuit.AddInitialVUser(initv_nodes, initv_values);
   circuit.AddCapacitorUser(e0, nodes0, capacitance);
   circuit.AddResistorUser (e1, nodes1, conductance);
   circuit.AddMeshUser     (e1);
   circuit.Setup();

   // Advance to t = 1 sec
   double dt = 0.01;
   double tfinal = 1.0;
   int nstep = int(tfinal/dt);
   double time = 0.0;
   for (int i=0;i<nstep;i++){
      time+=dt;
      circuit.Solve(dt,time);
   }

   double vna = circuit.GetMeshAnodeVoltage();
   double vnc = circuit.GetMeshCathodeVoltage();
   double vdrop = vnc - vna;

   double vdrop_expect = -initv_values[0]*(exp(-tfinal));

   double thresh = dt;

   EXPECT_TRUE(Omega_h::are_close(vdrop, vdrop_expect, thresh, 0.0));
}

TEST(circuit, RC_CHARGE) {

/*

YAML input:

circuit:
  mesh element: 0
  fixed:
    voltage nodes: [0,1]
    voltage values: [0.0,3.0]

  resistors:
    - {element: 0, nodes: [2,1], conductance: 0.1}
  capacitors:
    - {element: 1, nodes: [0,2], capacitance: 0.1}

*/

   std::vector<int>    fixedv_nodes  = {0,1};
   std::vector<double> fixedv_values = {0.0,3.0};

   int e0 = 0;
   std::vector<int> nodes0 = {1,2};
   double conductance = 0.1;

   int e1 = 1;
   std::vector<int> nodes1 = {2,0};
   double capacitance = 0.1;

   lgr::Circuit circuit;
   circuit.AddFixedVUser  (fixedv_nodes, fixedv_values);
   circuit.AddResistorUser (e0, nodes0, conductance);
   circuit.AddCapacitorUser(e1, nodes1, capacitance);
   circuit.AddMeshUser     (e0);
   circuit.Setup();

   // Advance to t = 1 sec
   double dt = 0.01;
   double tfinal = 1.0;
   int nstep = int(tfinal/dt);
   double time = 0.0;
   for (int i=0;i<nstep;i++){
      time+=dt;
      circuit.Solve(dt,time);
   }

   double vna = circuit.GetMeshAnodeVoltage();
   double vnc = circuit.GetMeshCathodeVoltage();
   double vdrop = vna - vnc;

   double vdrop_capacitor = fixedv_values[1]*(1.0-exp(-tfinal));
   double vdrop_expect    = fixedv_values[1] - vdrop_capacitor;

   double thresh = dt;

   EXPECT_TRUE(Omega_h::are_close(vdrop, vdrop_expect, thresh, 0.0));
}

TEST(circuit, RLC_UNDERDAMP) {

/*

YAML input:

circuit:
  mesh element: 1
  fixed:
    voltage nodes: [0]
    voltage values: [0.0]
  initial:
    voltage nodes: [1]
    voltage values: [10.0]

  capacitors:
    - {element: 0, nodes: [1,0], capacitance: 0.1111}
  resistors:
    - {element: 1, nodes: [2,1], conductance: 2.0}
  inductors:
    - {element: 2, nodes: [0,2], inductance: 1.0}

*/

   std::vector<int>    fixedv_nodes  = {0};
   std::vector<double> fixedv_values = {0.0};
   std::vector<int>    initv_nodes   = {1};
   std::vector<double> initv_values  = {10.0};

   int e0 = 0;
   std::vector<int> nodes0 = {1,0};
   double capacitance = 0.1111;

   int e1 = 1;
   std::vector<int> nodes1 = {2,1};
   double conductance = 2.0;

   int e2 = 2;
   std::vector<int> nodes2 = {0,2};
   double inductance = 1.0;

   lgr::Circuit circuit;
   circuit.AddFixedVUser  (fixedv_nodes, fixedv_values);
   circuit.AddInitialVUser(initv_nodes, initv_values);
   circuit.AddCapacitorUser(e0, nodes0, capacitance);
   circuit.AddResistorUser (e1, nodes1, conductance);
   circuit.AddInductorUser (e2, nodes2, inductance);
   circuit.AddMeshUser     (e1);
   circuit.Setup();

   // Advance to t = 2 sec
   double dt = 0.01;
   double tfinal = 2.0;
   int nstep = int(tfinal/dt);
   double time = 0.0;
   for (int i=0;i<nstep;i++){
      time+=dt;
      circuit.Solve(dt,time);
   }

   double vna = circuit.GetMeshAnodeVoltage();
   double vnc = circuit.GetMeshCathodeVoltage();
   double vdrop = vnc - vna;

   double alpha = (1.0/conductance)/(2.0*inductance);
   double omega = 1.0/sqrt(inductance*capacitance);
   double A     = sqrt(omega*omega - alpha*alpha);

   double current_expect = initv_values[0]/(inductance*A)*exp(-alpha*tfinal)*sin(A*tfinal);
   double vdrop_expect   = current_expect/conductance;

   double thresh = 4.0*dt; // Relax a bit

   EXPECT_TRUE(Omega_h::are_close(vdrop, vdrop_expect, thresh, 0.0));
}

TEST(circuit, RLC_OVERDAMP) {

/*

YAML input:

circuit:
  mesh element: 1
  fixed:
    voltage nodes: [0]
    voltage values: [0.0]
  initial:
    voltage nodes: [1]
    voltage values: [10.0]

  capacitors:
    - {element: 0, nodes: [1,0], capacitance: 0.1111}
  resistors:
    - {element: 1, nodes: [2,1], conductance: 0.1}
  inductors:
    - {element: 2, nodes: [0,2], inductance: 1.0}

*/

   std::vector<int>    fixedv_nodes  = {0};
   std::vector<double> fixedv_values = {0.0};
   std::vector<int>    initv_nodes   = {1};
   std::vector<double> initv_values  = {10.0};

   int e0 = 0;
   std::vector<int> nodes0 = {0,1};
   double capacitance = 0.1111;

   int e1 = 1;
   std::vector<int> nodes1 = {1,2};
   double conductance = 0.1;

   int e2 = 2;
   std::vector<int> nodes2 = {2,0};
   double inductance = 1.0;

   lgr::Circuit circuit;
   circuit.AddFixedVUser  (fixedv_nodes, fixedv_values);
   circuit.AddInitialVUser(initv_nodes, initv_values);
   circuit.AddCapacitorUser(e0, nodes0, capacitance);
   circuit.AddResistorUser (e1, nodes1, conductance);
   circuit.AddInductorUser (e2, nodes2, inductance);
   circuit.AddMeshUser     (e1);
   circuit.Setup();

   // Advance to t = 1 sec
   double dt = 0.01;
   double tfinal = 1.0;
   int nstep = int(tfinal/dt);
   double time = 0.0;
   for (int i=0;i<nstep;i++){
      time+=dt;
      circuit.Solve(dt,time);
   }

   double vna = circuit.GetMeshAnodeVoltage();
   double vnc = circuit.GetMeshCathodeVoltage();
   double vdrop = vna - vnc;

   double alpha = (1.0/conductance)/(2.0*inductance);
   double omega = 1.0/sqrt(inductance*capacitance);
   double A     = sqrt(alpha*alpha - omega*omega);

   double current_expect = initv_values[0]/(2.0*inductance*A)*exp(-alpha*tfinal)*(exp(A*tfinal)-exp(-A*tfinal));
   double vdrop_expect   = current_expect/conductance;

   double thresh = dt;

   EXPECT_TRUE(Omega_h::are_close(vdrop, vdrop_expect, thresh, 0.0));
}

TEST(circuit, R_DOUBLELOOP) {

/*

YAML input:

circuit:
  mesh element: 1
  fixed:
    voltage nodes: [0,1,3]
    voltage values: [0.0,32.0,20.0]

  resistors:
    - {element: 0, nodes: [2,1], conductance: 0.5}
    - {element: 1, nodes: [0,2], conductance: 0.125}
    - {element: 2, nodes: [2,3], conductance: 0.25}

*/

   std::vector<int>    fixedv_nodes  = {0,1,3};
   std::vector<double> fixedv_values = {0.0,32.0,20.0};

   int e0 = 0;
   int e1 = 1;
   int e2 = 2;
   std::vector<int> nodes0 = {2,1};
   std::vector<int> nodes1 = {0,2};
   std::vector<int> nodes2 = {2,3};
   double con0 = 0.5;
   double con1 = 0.125;
   double con2 = 0.25;

   lgr::Circuit circuit;
   circuit.AddFixedVUser  (fixedv_nodes, fixedv_values);
   circuit.AddResistorUser (e0, nodes0, con0);
   circuit.AddResistorUser (e1, nodes1, con1);
   circuit.AddResistorUser (e2, nodes2, con2);
   circuit.AddMeshUser     (e1);
   circuit.Setup();

   // Do not advance, but really doesn't matter since no C/L components
   double dt = 0.0;
   circuit.Solve(dt,0.0);

   double vn2 = circuit.GetMeshCathodeVoltage();
   double vexp_n2 = 24.0; // Hard coded for this circuit 

   // Compare voltage with analytic value with error of O(rthresh)
   double thresh = 1E-6;
   EXPECT_TRUE(Omega_h::are_close(vn2, vexp_n2, thresh, 0.0));

   // Changing conductance of mesh
   circuit.SetMeshConductance(1.0);

   // Do not advance, but really doesn't matter since no C/L components
   dt = 0.0;
   circuit.Solve(dt,0.0);

   vn2 = circuit.GetMeshCathodeVoltage();
   vexp_n2 = 12.0; // Hard coded for this new circuit 
   EXPECT_TRUE(Omega_h::are_close(vn2, vexp_n2, thresh, 0.0));
}

LGR_END_TESTS
