/*
//@HEADER
// ************************************************************************
//
//                        lgr v. 1.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  Glen A. Hansen (gahanse@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include "Driver.hpp"

#include <fstream>
#include <list>

#include "Fields.hpp"
#include "AdaptRecon.hpp"
#include "BoundaryCondition.hpp"
#include "ContactForce.hpp"
#include "LowRmPotentialSolve.hpp"
#include "LowRmRLCCircuit.hpp"
#include "CmdLineOutput.hpp"
#include "InitialConditions.hpp"
#include "ExplicitFunctors.hpp"
#include "MaterialModels.hpp"
#include "ConductivityModels.hpp"
#include "VizOutput.hpp"
#include "LagrangianTimeIntegration.hpp"
#include "ExactSolution.hpp"
#include "FieldDB.hpp"
#include "ErrorHandling.hpp"

#include <Omega_h_teuchos.hpp>

namespace lgr {

static bool is_restarting(Teuchos::ParameterList const &restart_pl) {
  return restart_pl.isType<double>("Start Time") ||
         restart_pl.isType<int>("Start Step");
}

// TODO: this function is over 500 lines long! Break it up!
template <int SpatialDim>
void run(
    Teuchos::ParameterList &                problem,
    const FEMesh<SpatialDim>&  mesh,
    MeshIO &                                mesh_io,
    std::string const&      viz_path,
    const int                               max_num_steps,
    const Scalar                            terminationTime) {

  Teuchos::ParameterList &time = problem.sublist(
      "Time", false, "All of the time parameters for time integration.");
  Teuchos::ParameterList &fieldData = problem.sublist(
      "Field Data", false,
      "All of the time parameters to define the FieldData.");
  Teuchos::ParameterList &initialCond =
      problem.sublist("Initial Conditions", false, "Initial conditions.");
  Teuchos::ParameterList &accel_bc_pl =
      problem.sublist("Boundary Conditions", false, "Boundary conditions.");
  Teuchos::ParameterList &traction_bc_pl =
      problem.sublist("Traction Boundary Conditions", false, "Traction boundary conditions.");
  Teuchos::ParameterList &viz_pl = problem.sublist("Visualization", false);
  Teuchos::ParameterList &restart_pl = problem.sublist("Restart", false);

  Teuchos::ParameterList &
  contact_params = problem.sublist("Contact", false, "Contact.");

  auto min_mass_density_allowed =
      problem.get<double>("Min Mass Density Allowed", 0.0);
  auto min_energy_density_allowed = problem.get<double>(
      "Min Internal Energy Density Allowed",
      Omega_h::ArithTraits<double>::min());

  mesh_io.computeSets();

  auto NumStates = time.get<int>("Number of States", 2);

  auto plot_cycle_frequency =
      viz_pl.get<int>("Step Period", std::numeric_limits<int>::max());
  auto plot_time_frequency =
      viz_pl.get<double>("Time Period", std::numeric_limits<double>::max());

  auto restart_cycle_period =
      restart_pl.get<int>("Step Period", std::numeric_limits<int>::max());
  auto restart_path = restart_pl.get<std::string>("Path", "restart");
  double restart_time = 0.0;
  int    restart_cycle = 0;
  auto should_load_restart = is_restarting(restart_pl);
  Omega_h::set_if_given(&restart_time, restart_pl, "Start Time");
  Omega_h::set_if_given(&restart_cycle, restart_pl, "Start Step");

  auto user_dt = time.get<double>("Fixed Time Step", 0.0);

  auto cfl = time.get<double>("Time Step Factor", 0.90);

  auto user_min_dt = time.get<double>("Minimum Time Step", 0.0);

  const comm::Machine machine = mesh.machine;

  //------------------------------------
  // Generate fields

  typedef Fields<SpatialDim> Fields;

  auto mesh_fields = Teuchos::rcp(new Fields(mesh, fieldData));

  AdaptRecon<SpatialDim> adaptRecon(problem, *mesh_fields, machine);

  VectorContributions<SpatialDim> accel_contribs;
  load_boundary_conditions(accel_contribs, accel_bc_pl);
  if (adaptRecon.isAdaptive() && !should_load_restart) {
    mark_fixed_velocity(accel_contribs, mesh_io);
  }
  VectorContributions<SpatialDim> internal_force_contribs;
  load_boundary_conditions(internal_force_contribs, traction_bc_pl);

  compute_contact_forces(internal_force_contribs, contact_params);

  auto cycle = restart_cycle;
  auto current_time = restart_time;

  accel_contribs.update(
      mesh_io.mesh_sets, current_time, mesh_fields->femesh.node_coords);
  internal_force_contribs.update(
      mesh_io.mesh_sets, current_time, mesh_fields->femesh.node_coords);
  //contact_contribs.update(
  //    mesh_io.mesh_sets, current_time, mesh_fields->femesh.node_coords);

  bool runLowRm = problem.isSublist("EM Physics");
  //  if (!runLowRm) std::cout << "'EM Physics' sublist not found.\n";
  Teuchos::RCP<LowRmPotentialSolve<SpatialDim>> potentialSolver;
  Teuchos::RCP<LowRmRLCCircuit>                 rlcCircuitSolver;
  typedef CrsLinearProblem<DefaultLocalOrdinal> LinearSolver;
  Teuchos::RCP<LinearSolver> linearSolver;

  double cgTol = 1e-12;
  int    cgMaxIters = 10000;
  Scalar timeIntervalForEMSolve =
      0;  // zero means we will do an EM solve at every time step (if we do one at all)
  Teuchos::RCP<std::ofstream>
      voltageDataFile;  // will be written to at same cadence as viz data

  // predeclare some things that get filled in during the low-Rm read, so that we can
  // reuse them when setting up post-adaptivity
  std::string outputPortNodeSetName = "", inputPortNodeSetName = "";
  std::map<std::string, std::string> voltageBCs;  // node set name --> value

  if (runLowRm) {
    voltageDataFile = Teuchos::rcp(new std::ofstream);
    if (!should_load_restart)  // if we *are* restarting, then we append to the voltage data file...
    {
      voltageDataFile->open("voltage_results.dat", std::ofstream::out);
      *voltageDataFile << "time\tV1\tV2\tV3\ti\tK11\tCum. Joules\n";
      voltageDataFile->close();
    }

    // ensure that conductivity exists and is correctly sized
    Kokkos::realloc(
        FieldDB<typename Fields::array_type>::Self()["conductivity"],
        mesh_fields->femesh.nelems);
    Kokkos::realloc(
        FieldDB<typename Fields::array_type>::Self()["element joule energy"],
        mesh_fields->femesh.nelems);
    Kokkos::realloc(
        FieldDB<typename Fields::array_type>::Self()["potential"],
        mesh_fields->femesh.nnodes);

    auto   lowRmParams = problem.sublist("EM Physics", true);
    Scalar V0 = lowRmParams.get<Scalar>("Initial Voltage");
    Scalar R = lowRmParams.get<Scalar>("R");
    Scalar L = lowRmParams.get<Scalar>("L");
    Scalar C = lowRmParams.get<Scalar>("C");
    int    m_series = lowRmParams.get<int>("M-Fold Series Symmetry", 1);
    int    m_parallel = lowRmParams.get<int>("M-Fold Parallel Symmetry", 1);
    timeIntervalForEMSolve = lowRmParams.get<Scalar>("Solve Interval");
    if (lowRmParams.isSublist("Input Port")) {
      inputPortNodeSetName =
          lowRmParams.sublist("Input Port").get<std::string>("Sides");
    }
    if (lowRmParams.isSublist("Output Port")) {
      outputPortNodeSetName =
          lowRmParams.sublist("Output Port").get<std::string>("Sides");
    }
    auto voltageBCParams = lowRmParams.sublist("Voltage BCs");
    for (Teuchos::ParameterList::ConstIterator i = voltageBCParams.begin();
         i != voltageBCParams.end(); i++) {
      const std::string &           name_i = voltageBCParams.name(i);
      const Teuchos::ParameterList &sublist = voltageBCParams.sublist(name_i);

      auto nodeSetName = sublist.get<std::string>("Sides");
      auto bcExpr = sublist.get<std::string>("Value Expression");

      voltageBCs.insert({nodeSetName, bcExpr});
    }

    using namespace std;
    cout << "Configured low-Rm physics with the following:\n";
    cout << "V0 = " << V0 << endl;
    cout << "R  = " << R << endl;
    cout << "L  = " << L << endl;
    cout << "C  = " << C << endl;
    cout << "Running with " << m_series << "-fold series symmetry\n";
    cout << "Running with " << m_parallel << "-fold parallel symmetry\n";

    cout << "Input Port set:  " << inputPortNodeSetName << endl;
    cout << "Output Port set: " << outputPortNodeSetName << endl;

    if (voltageBCs.size() > 0) {
      std::cout << "Voltage BCs:\n";
      for (auto entry : voltageBCs) {
        std::cout << "   " << entry.first << " --> " << entry.second
                  << std::endl;
      }
    }

    potentialSolver = Teuchos::rcp(
        new LowRmPotentialSolve<SpatialDim>(lowRmParams, mesh_fields, machine));
    potentialSolver->setUseMFoldSymmetry(m_series, m_parallel);

    potentialSolver->setPorts(
        mesh_io, inputPortNodeSetName, outputPortNodeSetName);

    for (auto entry : voltageBCs) {
      auto &nodesets = mesh_io.mesh_sets[Omega_h::NODE_SET];

      auto nodeSetName = entry.first;
      auto bcExpr = entry.second;

      auto nsInputIter = nodesets.find(nodeSetName);
      LGR_THROW_IF(
          nsInputIter == nodesets.end(),
          "node set " << nodeSetName << " doesn't exist!\n");
      auto localOrdinals = nsInputIter->second;

      potentialSolver->setBC(
          bcExpr, localOrdinals, true);  // true: add to any existing BCs...
    }

    potentialSolver->setConductivity(Conductivity<Fields>());

    if (timeIntervalForEMSolve == 0)
      cout << "Will solve the low Rm problem at each time step.\n";
    else
      cout << "Will solve the low Rm problem every " << timeIntervalForEMSolve
           << " simulation seconds.\n";
    rlcCircuitSolver = Teuchos::rcp(new LowRmRLCCircuit(R, L, C, V0));
  }

  if (0 == comm::rank(machine)) printHeaders();

  int current_state = cycle % NumStates;
  int next_state = cycle % NumStates;

  //------------------------------------
  // Initialization
  if (should_load_restart) {
    if (comm::rank(machine) == 0) {
      std::cout << "Restarting solution from file...\n";
    }
    adaptRecon.loadRestart(mesh_io);
    if (runLowRm) {
      std::ifstream     voltageRestart;
      std::stringstream vss;
      vss << restart_path << '_' << restart_cycle << "_voltage.dat";
      voltageRestart.open(vss.str(), std::ofstream::in);

      double v1, v2, v3, v4, i;
      voltageRestart >> v1;
      voltageRestart >> v2;
      voltageRestart >> v3;
      voltageRestart >> v4;
      voltageRestart >> i;
      voltageRestart.close();

      rlcCircuitSolver->setValues(v1, v2, v3, v4, i);
    }
  } else {
    InitialConditions<Fields> ic(initialCond);

    ic.set(
        mesh_io.mesh_sets[Omega_h::NODE_SET], Velocity<Fields>(),
        Displacement<Fields>(), mesh_fields->femesh);
    ic.set(
        mesh_io.mesh_sets[Omega_h::ELEM_SET], MassDensity<Fields>(),
        InternalEnergyPerUnitMass<Fields>(), mesh_fields->femesh);
    ic.set(
        mesh_io.mesh_sets[Omega_h::SIDE_SET], 
        mesh_io.mesh_sets[Omega_h::ELEM_SET],
        MagneticFaceFlux<Fields>(), mesh_fields->femesh);

    initialize_element<SpatialDim>::apply(
        *mesh_fields, ic);

    initialize_node<SpatialDim>::apply(*mesh_fields);
  }

  // material(s) for the problem
  std::list<
      std::shared_ptr<MaterialModelBase<SpatialDim>>>
      theMaterialModels;
  {
    auto &materialModelParameterList =
        problem.sublist("Material Models", false);
    createMaterialModels(
        materialModelParameterList, *mesh_fields,
        mesh_io.mesh_sets[Omega_h::ELEM_SET], theMaterialModels);
  }
  for (auto matPtr : theMaterialModels) {
    matPtr->initializeElements(*mesh_fields);
  }

  // conductivity(s) for the problem
  std::list<std::shared_ptr<
      ConductivityModelBase<SpatialDim>>>
      theConductivityModels;
  {
    if (runLowRm) {
      auto &ConductivityModelParameterList =
          problem.sublist("Electrical Conductivity Models", false);
      createConductivityModels(
          ConductivityModelParameterList, *mesh_fields,
          mesh_io.mesh_sets[Omega_h::ELEM_SET], theConductivityModels);

      for (auto matPtr : theConductivityModels) {
        matPtr->initializeElements(*mesh_fields);
      }
    }
  }

  if (adaptRecon.isAdaptive()) {
    adaptRecon.computeMetric(current_state, next_state);
  }

  check_densities(
      machine, *mesh_fields, next_state, min_mass_density_allowed,
      min_energy_density_allowed);

  //--------------------------------------------------------------------------
  // We will call a sequence of functions.  These functions have been
  // grouped into several functors to balance the number of global memory
  // accesses versus requiring too many registers or too much L1 cache.
  // Global memory accees have read/write cost and memory subsystem contention cost.
  //--------------------------------------------------------------------------

  auto plot_viz_time = plot_time_frequency;

  mesh_fields->conformGeom("vel", Fields::getGeomFromSA(Velocity<Fields>(), 0));
  mesh_fields->conform("nodal_mass", NodalMass<Fields>());

  Scalar dt(0.0);
  {
    const Scalar alpha(1.0);
    //calculate initial time step; this uses the velocity gradient.
    grad<SpatialDim>::apply(
        *mesh_fields, current_state, next_state, alpha);

    explicit_time_step<SpatialDim> timeStepCalculator(
        *mesh_fields);
    dt = timeStepCalculator.apply(current_state);
    dt = comm::min(machine, dt);
    dt *= cfl;

    // quit if user's minimum dt reached
    LGR_THROW_IF(
        ((comm::rank(machine) == 0) && (dt < user_min_dt)),
        "*********************** USER MESSAGE ************************\n"
            << "User requested minimum time step of " << user_min_dt
            << " greater than calculated stable time step of " << dt << ".\n "
            << "Simulation shutting down.\n"
            << "*************************************************************"
               "\n");

    // force fixed time step if input
    if (user_dt > 0.0) {
      if (user_dt > dt && !comm::rank(machine)) {
        std::cout << "WARNING: OVERRIDING MAX STABLE TIME STEP " << dt
                  << " WITH " << user_dt << '\n';
      }
      dt = user_dt;
    }
  }

  VizOutput viz_output(mesh_io.getMesh(), viz_path, viz_pl, restart_time);

  //write out initial data
  {
    GlobalTallies<SpatialDim> globalTally(
        *mesh_fields, current_state);
    globalTally.apply();
    const std::vector<double> &localTallies =
        globalTally.contiguousMemoryTallies;
    std::vector<double> globalTallies(localTallies);
    comm::allReduce(
        machine, localTallies.size(), localTallies.data(),
        globalTallies.data());
    if (0 == comm::rank(machine)) {
      printGlobals(
          cycle, current_time, dt, globalTallies.size(), globalTallies.data());
    }

    viz_output.writeOutputFile(
        *mesh_fields, cycle, current_state, current_time);

    if (runLowRm) {
      voltageDataFile->open(
          "voltage_results.dat", std::ofstream::out | std::ofstream::app);
      *voltageDataFile << current_time << "\t";
      *voltageDataFile << rlcCircuitSolver->v1() << "\t";
      *voltageDataFile << rlcCircuitSolver->v2() << "\t";
      *voltageDataFile << rlcCircuitSolver->v3() << "\t";
      *voltageDataFile << rlcCircuitSolver->i() << "\t";
      *voltageDataFile << 0.0
                       << "\t";  // getK11() won't work until we have solved...
      *voltageDataFile << potentialSolver->getTotalJoulesAdded() << "\n";
      voltageDataFile->close();
    }
  }

  LagrangianStep<SpatialDim> lagrangianStep(
      theMaterialModels, *mesh_fields, machine, mesh_io.getMesh());

  if (runLowRm) {
    potentialSolver->initialize();
  }

  Scalar lastEMSolveTime = -1e12;
  while ((cycle < max_num_steps) && (current_time < terminationTime)) {

    //cycle the states
    current_state = next_state;
    ++next_state;
    next_state %= NumStates;

    lagrangianStep.advanceTime(
        accel_contribs,
        internal_force_contribs,
        current_time,
        dt,
        current_state,
        next_state);

    check_densities(
        machine, *mesh_fields, next_state, min_mass_density_allowed,
        min_energy_density_allowed);

    if (runLowRm) {
      if (current_time > lastEMSolveTime + timeIntervalForEMSolve) {
        for (auto conductivityModelPtr : theConductivityModels)
          conductivityModelPtr->updateElements(*mesh_fields, next_state);

        potentialSolver->setConductivity(
            Conductivity<Fields>());  // probably this is redundant
        potentialSolver->assemble();
        // for now, defensive programming: force recreation of linear solver
        // (I'm a bit suspicious of the fact that the residual is *exactly* the same each time...
        //  There may be something we need to do to get AmgX to refresh its data...
        //  )
        linearSolver = Teuchos::null;
        if (linearSolver == Teuchos::null) {
          linearSolver = potentialSolver->getDefaultSolver(cgTol, cgMaxIters);
          // quit if no linear solver is available
          LGR_THROW_IF(((comm::rank(machine) == 0) && (linearSolver == Teuchos::null)),
                         "***************************** USER MESSAGE ************************\n"
                         << "No linear solver is available, and low-Rm was requested.  Exiting…\n"
                         << "*******************************************************************\n");
        } else {
          // TODO: give the linear solver a chance to update the matrix here and/or recompute the preconditioner
          // (ViennaCL, e.g., has its own copy of the matrix; we'll need to copy the newly-assembled matrix there.)
          linearSolver->initializeSolver();
        }
        linearSolver->solve();
        lastEMSolveTime = current_time;
      }
      auto element_internal_energy = ElementInternalEnergy<Fields>();
      auto element_joule_energy = ElementJouleEnergy<Fields>();
      auto element_mass = ElementMass<Fields>();
      auto element_volume = ElementVolume<Fields>();
      auto internal_energy_per_unit_mass = InternalEnergyPerUnitMass<Fields>();
      auto internal_energy_density = InternalEnergyDensity<Fields>();

      potentialSolver->determineJouleHeating(
          element_internal_energy, element_joule_energy, rlcCircuitSolver->v3(),
          dt);  // computes K11, deposits energy

      // update the internal energy density and the internal energy per unit mass to be consistent with this internal energy
      Kokkos::parallel_for(
          Kokkos::RangePolicy<int>(0, mesh_fields->femesh.nelems),
          LAMBDA_EXPRESSION(int ielem) {
            internal_energy_per_unit_mass(ielem, next_state) =
                element_internal_energy(ielem) / element_mass(ielem);
            internal_energy_density(ielem) =
								element_internal_energy(ielem) / element_volume(ielem);
          },
          "Update internal energy density and energy per unit mass to be "
          "consistent with Joule heating");

      Scalar K11 = potentialSolver->getK11();
      rlcCircuitSolver->setK11(K11);
      rlcCircuitSolver->takeTimeStep(dt);

      using namespace std;
      cout << "Voltages: V1 = " << rlcCircuitSolver->v1() << "; ";
      cout << "V2 = " << rlcCircuitSolver->v2() << "; ";
      cout << "V3 = " << rlcCircuitSolver->v3() << " (K11 = " << K11 << "; "
           << potentialSolver->getTotalJoulesAdded()
           << " total Joules added)\n";
    }

    // Do adaptivity calculations
    auto mesh_adapted =
        adaptRecon.adaptMeshAndRemapFields(mesh_io, current_state, next_state);

    check_densities(
        machine, *mesh_fields, next_state, min_mass_density_allowed,
        min_energy_density_allowed);

    if (mesh_adapted) {
      if (runLowRm) {
        // ensure that conductivity is correctly sized
        Kokkos::realloc(
            FieldDB<typename Fields::array_type>::Self()["conductivity"],
            mesh_fields->femesh.nelems);
        Kokkos::realloc(
            FieldDB<typename Fields::array_type>::Self()["element joule energy"],
            mesh_fields->femesh.nelems);
        Kokkos::realloc(
            FieldDB<typename Fields::array_type>::Self()["potential"],
            mesh_fields->femesh.nnodes);

        potentialSolver->resetMesh(mesh_fields);  // also clears BCs
        potentialSolver->setPorts(
            mesh_io, inputPortNodeSetName, outputPortNodeSetName);

        for (auto entry : voltageBCs) {
          auto &nodesets = mesh_io.mesh_sets[Omega_h::NODE_SET];

          auto nodeSetName = entry.first;
          auto bcExpr = entry.second;

          auto nsInputIter = nodesets.find(nodeSetName);
          LGR_THROW_IF(
              nsInputIter == nodesets.end(),
              "node set " << nodeSetName << " doesn't exist!\n");
          auto localOrdinals = nsInputIter->second;

          potentialSolver->setBC(
              bcExpr, localOrdinals, true);  // true: add to any existing BCs...
        }
      }
      //re-create material(s) for the problem
      theMaterialModels.clear();
      auto &material_pl = problem.sublist("Material Models", false);
      createMaterialModels(
          material_pl, *mesh_fields, mesh_io.mesh_sets[Omega_h::ELEM_SET],
          theMaterialModels);
      for (auto matPtr : theMaterialModels) {
        matPtr->updateElements(*mesh_fields, next_state, current_time, dt);
      }
      if (runLowRm) {
        theConductivityModels.clear();
        auto &ConductivityModelParameterList =
            problem.sublist("Electrical Conductivity Models", false);
        createConductivityModels(
            ConductivityModelParameterList, *mesh_fields,
            mesh_io.mesh_sets[Omega_h::ELEM_SET], theConductivityModels);
        // we can wait to call updateElements(); this will happen immediately before the next low-Rm solve...

        potentialSolver->setConductivity(
            Conductivity<Fields>());  // probably this is redundant
        potentialSolver
            ->initialize();  // reconstruct stiffness matrix, RHS and solution vectors
        linearSolver = Teuchos::
            null;  // force reconstruction of linear solver after next assembly
      }
    }

    current_time += dt;
    ++cycle;

    //update boundary condition node lists
    accel_contribs.update(
        mesh_io.mesh_sets, current_time, mesh_fields->femesh.node_coords);
    internal_force_contribs.update(
        mesh_io.mesh_sets, current_time, mesh_fields->femesh.node_coords);

    GlobalTallies<SpatialDim> globalTally(
        *mesh_fields, next_state);
    globalTally.apply();
    auto &              localTallies = globalTally.contiguousMemoryTallies;
    std::vector<double> globalTallies(localTallies);
    comm::allReduce(
        machine, localTallies.size(), localTallies.data(),
        globalTallies.data());
    if (0 == comm::rank(machine)) {
      printGlobals(
          cycle, current_time, dt, globalTallies.size(), globalTallies.data());
    }
    //calculate next time step; this uses the velocity gradient.
    {
      /* TODO: consolidate this with the exact same logic for the first time step */
      explicit_time_step<SpatialDim> timeStepCalculator(*mesh_fields);
      dt = timeStepCalculator.apply(next_state);
      dt = comm::min(machine, dt);
      dt *= cfl;

      // quit if user's minimum dt reached
      LGR_THROW_IF(
          ((comm::rank(machine) == 0) && (dt < user_min_dt)),
          "*********************** USER MESSAGE ************************\n"
              << "User requested minimum time step of " << user_min_dt
              << " greater than calculated stable time step of " << dt << ".\n "
              << "Simulation shutting down.\n"
              << "*************************************************************"
                 "\n");

      // force fixed time step if input
      if (user_dt > 0.0) {
        if (user_dt > dt && !comm::rank(machine)) {
          std::cout << "WARNING: OVERRIDING STABLE TIME STEP " << dt << " WITH "
                    << user_dt << '\n';
        }
        dt = user_dt;
      }
    }

    bool viz_triggered = false;
    if (0 == cycle % plot_cycle_frequency) viz_triggered = true;
    if (current_time >= plot_viz_time) {
      viz_triggered = true;
      plot_viz_time += plot_time_frequency;
    }
    if (viz_triggered) {
      if (runLowRm) {
        // write voltage values to file
        if (comm::rank(mesh_io.getMachine()) ==
            0)  // right now, we only support low-Rm on a single MPI rank.  But that may change...
        {
          voltageDataFile->open(
              "voltage_results.dat", std::ofstream::out | std::ofstream::app);
          *voltageDataFile << current_time << "\t";
          *voltageDataFile << rlcCircuitSolver->v1() << "\t";
          *voltageDataFile << rlcCircuitSolver->v2() << "\t";
          *voltageDataFile << rlcCircuitSolver->v3() << "\t";
          *voltageDataFile << rlcCircuitSolver->i() << "\t";
          *voltageDataFile << potentialSolver->getK11() << "\t";
          *voltageDataFile << potentialSolver->getTotalJoulesAdded() << "\n";
          voltageDataFile->close();
        }
        // copy the solution from low-Rm to the nodal potential store in the Field DB
        // the LHS has a different shape than the ElectricPotential store, anticipating the possibility of multiple simultaneous RHSes, similar
        // to the Alegra FE approach to low-Rm (which allows arbitrary circuits, not just RLC).
        // To accommodate this, we take a subview.  (This actually has all the data.)

        auto subview = Kokkos::subview(
            potentialSolver->getLHS(), 0,
            Kokkos::ALL());  // solveIndex, rowIndex
        Kokkos::deep_copy(ElectricPotential<Fields>(), subview);
      }
      viz_output.writeOutputFile(
          *mesh_fields, cycle, next_state, current_time);
    }

    if (0 == cycle % restart_cycle_period) {
      if (0 == comm::rank(machine)) {
        std::cout << "Restart Dump | cycle: " << cycle << " time: ";
        auto precision_before = std::cout.precision();
        std::ios::fmtflags stream_state(std::cout.flags());
        std::cout << std::scientific << std::setprecision(18) << current_time;
        std::cout.flags(stream_state);
        std::cout.precision(precision_before);
      }
      std::stringstream ss;
      ss << restart_path << '_' << cycle << ".osh";
      auto s = ss.str();
      adaptRecon.writeRestart(s, next_state);

      if (runLowRm) {
        // then also write a restart file for RLC state
        std::ofstream     voltageRestart;
        std::stringstream vss;
        vss << restart_path << '_' << cycle << "_voltage.dat";
        voltageRestart.open(vss.str(), std::ofstream::out);
        voltageRestart << rlcCircuitSolver->v1() << "\t";
        voltageRestart << rlcCircuitSolver->v2() << "\t";
        voltageRestart << rlcCircuitSolver->v3() << "\t";
        voltageRestart << rlcCircuitSolver->v4() << "\t";
        voltageRestart << rlcCircuitSolver->i() << "\n";
        voltageRestart.close();
      }
    }

  }  //end while ( (step<max_num_steps) && (current_time<terminationTime) )

  if (problem.isSublist("Scatterplots")) {
    auto &sps_pl = problem.sublist("Scatterplots");
    for (auto it = sps_pl.begin(), end = sps_pl.end(); it != end; ++it) {
      auto  sp_name = sps_pl.name(it);
      auto &sp_pl = sps_pl.sublist(sp_name);
      int   ent_dim = 0;
      auto  m = mesh_fields->femesh.omega_h_mesh;
      if (sp_pl.isType<std::string>("Entity")) {
        ent_dim =
            Omega_h::get_ent_dim_by_name(m, sp_pl.get<std::string>("Entity"));
      }
      auto            field_name = sp_pl.get<std::string>("Field");
      Omega_h::TagSet tags;
      tags[std::size_t(ent_dim)].insert(field_name);
      mesh_fields->copyTagsToMesh(tags, next_state);
      Omega_h::write_scatterplot(m, sp_pl);
      mesh_fields->cleanTagsFromMesh(tags);
    } // end of each scatterplot
  } // end of scatterplots

  if (problem.isSublist("ExactSolution")) {
    const Teuchos::ParameterList& params = problem.sublist("ExactSolution");
    ExactSolution<SpatialDim> (params, mesh_fields, next_state, machine);
  } 
}

template <int SpatialDim>
void driver(
    Omega_h::Library *      lib_osh,
    Teuchos::ParameterList &problem,
    comm::Machine           machine,
    const std::string&       input_filename,
    const std::string&       viz_path) {

  typedef MeshFixture<SpatialDim> fixture_type;

  typedef typename fixture_type::FEMeshType mesh_type;

  Teuchos::ParameterList &time = problem.sublist(
      "Time", false, "All of the time parameters for time integration.");
  const int max_num_steps =
      time.get<int>("Steps", std::numeric_limits<int>::max());
  const Scalar terminationTime =
      time.get<double>("Termination Time", std::numeric_limits<double>::max());

  auto &assoc_pl = problem.sublist("Associations");

  MeshIO mesh_io(
      lib_osh, input_filename, assoc_pl, machine,
      is_restarting(time));

  {
    // if the user requested it, run the quality-fixer on the mesh
    bool fixQuality = problem.get<bool>("Fix Mesh Quality", false);
    if (fixQuality) {
      auto               omegaHMesh = mesh_io.getMesh();
      Omega_h::AdaptOpts opts(omegaHMesh);
      Omega_h::fix(
          omegaHMesh, opts, OMEGA_H_ISO_LENGTH, true);  // true: verbose

      omegaHMesh->set_parting(OMEGA_H_GHOSTED);
    }
  }

  mesh_type mesh = fixture_type::create(mesh_io, machine);

  if (comm::rank(machine) == 0) std::cout << "\nlgr!\n";

  run<SpatialDim>(
      problem, mesh, mesh_io, viz_path, max_num_steps, terminationTime);
}

void driver(
    Omega_h::Library *      lib_osh,
    Teuchos::ParameterList &problem,
    comm::Machine           machine,
    const std::string&       input_filename,
    const std::string&       viz_path) {
  const int spaceDim = problem.get<int>("Spatial Dimension", 3);

  if (spaceDim == 3) {
      driver<3>(lib_osh, problem, machine, input_filename, viz_path);
      lgr::FieldDB_Finalize<3>();
  } else if (spaceDim == 2) {
      driver<2>(lib_osh, problem, machine, input_filename, viz_path);
      lgr::FieldDB_Finalize<2>();
  } else {
      std::cout << "Unhandled Spatial Dimension = " << spaceDim << std::endl;
  }
}

#define LGR_EXPL_INST(SpatialDim) \
template \
void run( \
    Teuchos::ParameterList &                problem, \
    const FEMesh<SpatialDim>&  mesh, \
    MeshIO &                                mesh_io, \
    std::string const&      viz_path, \
    const int                               max_num_steps, \
    const Scalar                            terminationTime); \
template \
void driver<SpatialDim>( \
    Omega_h::Library *      lib_osh, \
    Teuchos::ParameterList &problem, \
    comm::Machine           machine, \
    const std::string&       input_filename, \
    const std::string&       viz_path);
LGR_EXPL_INST(3)
LGR_EXPL_INST(2)
#undef LGR_EXPL_INST

}  // namespace lgr
