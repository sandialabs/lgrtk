#include <lgr_circuit.hpp>
#include <Omega_h_scalar.hpp>
#include "lgr_gtest.hpp"
#include <cmath>
#include <vector>

TEST(circuit, RC) {

   /*
      Circuit setup:
        * If read in from YAML file we would have:
     
            circuit:
                capacitors:
                  - 
                    element: 0
                    nodes: [0,1]
                    capacitance: 0.1
                    initial voltage: [3,0]
           
                resistors:
                  - 
                    element: 1
                    nodes: [1,0]
                    conductance: 0.1
                  
        * Instead we setup with User calls for testing
     
        * Corresponding circuit is:
     
                        Node 1
              --....-- --------
             |        |        |
             | 3 V    | 0.1 F  / 
             -       ---       \
            ---      ---       / 10 ohm
             |        |  (CW)  \
             |        |        |
              -------- --------
                        Node 0
     
        * Prior to simulation, swtich (....) is closed allowing
          capacitor to charge
     
        * When simulation begins, switch is opened and capacitor
          discharges
     
        * Simulation output is compared to voltage at node 1
   */
   int e0 = 0;
   std::vector<int> nodesr = {1,0};
   double conductance = 0.1;

   int e1 = 1;
   std::vector<int> nodesc = {0,1};
   double capacitance = 0.1;
   double v0val = 3.0;
   std::vector<double> v0 = {0.0,v0val};

   lgr::Circuit c;
   c.AddResistorUser (e0, nodesr, conductance);
   c.AddCapacitorUser(e1, nodesc, capacitance, v0);
   c.Setup();

   // Advance in time to 1 sec (i.e., 1 tau since R*C = 1)
   double dt = 0.01;
   double tfinal = 1.0;
   int nstep = int(tfinal/dt);
   for (int i=0;i<nstep;i++){
      c.Solve(dt);
   }

   // Compare voltage with analytic value with error of O(dt)
   double vn1 = c.GetNodeVoltage(1);
   double vn1_expect = v0val*exp(-tfinal*conductance/capacitance);
   EXPECT_TRUE(Omega_h::are_close(vn1, vn1_expect, dt, 0.0));

}

TEST(circuit, RSerial) {

   /*
    Circuit setup:
      * If read in from YAML file we would have:
      
        circuit:
        
            resistors:
              - 
                element: 0
                nodes: [0,1]
                conductance: 1E-2
              - 
                element: 1
                nodes: [1,2]
                conductance: 1E-3
              - 
                element: 2
                nodes: [2,3]
                conductance: 1E-4
        
            voltage sources:
              - 
                element: 3
                nodes: [3,0]
                voltage: [5,0]
   
      * Instead we setup with User calls for testing
   
      * Corresponding circuit is:
   
      Node 0   Node 1   Node 2   Node 3
         |        |        |       |
         ----/\/\----/\/\----/\/\----
        |    100     1000    10000   |
        |    ohm     ohm     ohm     |
        |                            |
        |  (CW)                      |
        |              | 5 V         |
         ------------| |-------------
                       |
   
      * Simulation output is compared to expected voltage drops
        across each resistor
   */
   int e0 = 0;
   int e1 = 1;
   int e2 = 2;
   std::vector<int> nodesr0 = {0,1};
   std::vector<int> nodesr1 = {1,2};
   std::vector<int> nodesr2 = {2,3};
   double con0 = 1E-2;
   double con1 = 1E-3;
   double con2 = 1E-4;

   int e3 = 3;
   std::vector<int> nodesv0 = {3,0};
   std::vector<double> vsource = {5,0};

   lgr::Circuit c;
   c.AddResistorUser (e0, nodesr0, con0);
   c.AddResistorUser (e1, nodesr1, con1);
   c.AddResistorUser (e2, nodesr2, con2);
   c.AddVSourceUser  (e3, nodesv0, vsource);
   c.Setup();

   double vdrop_r0 = c.GetNodeVoltage(1) - c.GetNodeVoltage(0);
   double vdrop_r1 = c.GetNodeVoltage(2) - c.GetNodeVoltage(1);
   double vdrop_r2 = c.GetNodeVoltage(3) - c.GetNodeVoltage(2);

   double fac = vsource[0]/(1.0/con0 + 1.0/con1 + 1.0/con2);
   double vexp_r0 = fac/con0;
   double vexp_r1 = fac/con1;
   double vexp_r2 = fac/con2;

   // Compare voltage with analytic value with error of O(rthresh)
   double rthresh = 1E-6;
   EXPECT_TRUE(Omega_h::are_close(vdrop_r0, vexp_r0, rthresh, 0.0));
   EXPECT_TRUE(Omega_h::are_close(vdrop_r1, vexp_r1, rthresh, 0.0));
   EXPECT_TRUE(Omega_h::are_close(vdrop_r2, vexp_r2, rthresh, 0.0));

}

TEST(circuit, RDoubleLoop) {

   /*
    Circuit setup:
      * If read in from YAML file we would have:
      
        circuit:

             mesh element: 1
        
             resistors:
               - 
                 element: 0
                 nodes: [1,2]
                 conductance: 0.5
               - 
                 element: 1
                 nodes: [2,0]
                 conductance: 0.125
               - 
                 element: 2
                 nodes: [3,2]
                 conductance: 0.25
         
             voltage sources:
               - 
                 element: 3
                 nodes: [0,1]
                 voltage: [0,32]
               - 
                 element: 4
                 nodes: [0,3]
                 voltage: [0,20]
    
      * Instead we setup with User calls for testing
   
      * Corresponding circuit is:
   
                     Node 2
                       |
                 2     |   4
        Node 1  ohm    |  ohm    Node 3
              --/\/\--- --/\/\---
             |         |         |
             | 32 V    /         | 20 V
            ---        \ 8      ---
             -         / ohm     -
             | (CW)    \         |
             |         |  (CCW)  |
              --------- ---------
                    Node 0
   
      * Simulation output is compared to expected voltage
        at node 2
   */
   int e0 = 0;
   int e1 = 1;
   int e2 = 2;
   std::vector<int> nodesr0 = {1,2};
   std::vector<int> nodesr1 = {2,0};
   std::vector<int> nodesr2 = {3,2};
   double con0 = 0.5;
   double con1 = 0.125;
   double con2 = 0.25;

   int e3 = 3;
   int e4 = 4;
   std::vector<int> nodesv0 = {0,1};
   std::vector<int> nodesv1 = {0,3};
   std::vector<double> vsource0 = {0,32};
   std::vector<double> vsource1 = {0,20};

   lgr::Circuit c;
   c.AddMeshUser     (e1);
   c.AddResistorUser (e0, nodesr0, con0);
   c.AddResistorUser (e1, nodesr1, con1);
   c.AddResistorUser (e2, nodesr2, con2);
   c.AddVSourceUser  (e3, nodesv0, vsource0);
   c.AddVSourceUser  (e4, nodesv1, vsource1);
   c.Setup();

   double vn2 = c.GetNodeVoltage(2);
   double vexp_n2 = 24.0; // Hard coded for this circuit 

   // Compare voltage with analytic value with error of O(rthresh)
   double rthresh = 1E-6;
   EXPECT_TRUE(Omega_h::are_close(vn2, vexp_n2, rthresh, 0.0));

   // Let us also test if we can edit conductance of 8 ohm resistor,
   // so we will change it to 1 ohm. Note that since we set the key
   // 'mesh element' at input to e1, we can simply set the Mesh conductance
   // as follows
   c.SetMeshConductance(1.0);
   c.Setup();

   vn2 = c.GetMeshCathodeVoltage();
   vexp_n2 = 12.0; // Hard coded for this new circuit 
   EXPECT_TRUE(Omega_h::are_close(vn2, vexp_n2, rthresh, 0.0));

}

LGR_END_TESTS
