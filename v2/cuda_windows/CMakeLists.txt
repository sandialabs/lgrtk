cmake_minimum_required(VERSION 3.9)
# Compile LGR with CUDA on Windows
# Tested with:
#    Visual Studio Professional 2019 (Version 16.4.0)
#    CUDA 10.2 (Version 10.2.89_441.22)
#    NVIDIA Quadro P2000 GPU
#
# To run:
#    * Download all source files from omega_h/src and put them in src/ (execpt the CMakeLists.txt)
#    * Download all source files from lgr/v2/src and put them in src/ (execpt the CMakeLists.txt)

project(lgr)

# Old cmake cuda
find_package(CUDA REQUIRED)

# Find omega_h files
file(GLOB OMEGA_H_CUDA_FILES 
  src/Omega_h_adapt.cpp
  src/Omega_h_adj.cpp
  src/Omega_h_align.cpp
  src/Omega_h_amr.cpp
  src/Omega_h_amr_topology.cpp
  src/Omega_h_amr_transfer.cpp
  src/Omega_h_any.cpp
  src/Omega_h_approach.cpp
  src/Omega_h_array.cpp
  src/Omega_h_array_ops.cpp
  src/Omega_h_assoc.cpp
  src/Omega_h_base64.cpp
  src/Omega_h_bbox.cpp
  src/Omega_h_bcast.cpp
  src/Omega_h_bipart.cpp
  src/Omega_h_box.cpp
  src/Omega_h_build.cpp
  src/Omega_h_build_parser.cpp
  src/Omega_h_chartab.cpp
  src/Omega_h_check_collapse.cpp
  src/Omega_h_class.cpp
  src/Omega_h_cmdline.cpp
  src/Omega_h_coarsen.cpp
  src/Omega_h_coarsen_overshoot.cpp
  src/Omega_h_coarsen_qualities.cpp
  src/Omega_h_coarsen_topology.cpp
  src/Omega_h_collapse_rail.cpp
  src/Omega_h_comm.cpp
  src/Omega_h_compare.cpp
  src/Omega_h_confined.cpp
  src/Omega_h_conserve.cpp
  src/Omega_h_dist.cpp
  src/Omega_h_eigen.cpp
  src/Omega_h_expr.cpp
  src/Omega_h_fail.cpp
  src/Omega_h_fence.cpp
  src/Omega_h_file.cpp
  src/Omega_h_filesystem.cpp
  src/Omega_h_finite_automaton.cpp
  src/Omega_h_fix.cpp
  src/Omega_h_future.cpp
  src/Omega_h_ghost.cpp
  src/Omega_h_globals.cpp
  src/Omega_h_gmsh.cpp
  src/Omega_h_grammar.cpp
  src/Omega_h_graph.cpp
  src/Omega_h_hilbert.cpp
  src/Omega_h_histogram.cpp
  src/Omega_h_indset.cpp
  src/Omega_h_inertia.cpp
  src/Omega_h_input.cpp
  src/Omega_h_int128.cpp
  src/Omega_h_int_scan.cpp
  src/Omega_h_language.cpp
  src/Omega_h_laplace.cpp
  src/Omega_h_library.cpp
  src/Omega_h_linpart.cpp
  src/Omega_h_malloc.cpp
  src/Omega_h_map.cpp
  src/Omega_h_mark.cpp
  src/Omega_h_math_lang.cpp
  src/Omega_h_matrix.cpp
  src/Omega_h_mesh.cpp
  src/Omega_h_metric.cpp
  src/Omega_h_metric_input.cpp
  src/Omega_h_migrate.cpp
  src/Omega_h_modify.cpp
  src/Omega_h_owners.cpp
  src/Omega_h_parser.cpp
  src/Omega_h_parser_graph.cpp
  src/Omega_h_pool.cpp
  src/Omega_h_print.cpp
  src/Omega_h_profile.cpp
  src/Omega_h_quality.cpp
  src/Omega_h_reader.cpp
  src/Omega_h_recover.cpp
  src/Omega_h_refine.cpp
  src/Omega_h_refine_qualities.cpp
  src/Omega_h_refine_topology.cpp
  src/Omega_h_regex.cpp
  src/Omega_h_regress.cpp
  src/Omega_h_remotes.cpp
  src/Omega_h_reorder.cpp
  src/Omega_h_scatterplot.cpp
  src/Omega_h_shape.cpp
  src/Omega_h_shared_alloc.cpp
  src/Omega_h_simplify.cpp
  src/Omega_h_sort.cpp
  src/Omega_h_surface.cpp
  src/Omega_h_swap.cpp
  src/Omega_h_swap2d.cpp
  src/Omega_h_swap2d_qualities.cpp
  src/Omega_h_swap2d_topology.cpp
  src/Omega_h_swap3d.cpp
  src/Omega_h_swap3d_qualities.cpp
  src/Omega_h_swap3d_topology.cpp
  src/Omega_h_tag.cpp
  src/Omega_h_timer.cpp
  src/Omega_h_transfer.cpp
  src/Omega_h_unmap_mesh.cpp
  src/Omega_h_vector.cpp
  src/Omega_h_vtk.cpp
  src/Omega_h_xml.cpp
  src/Omega_h_xml_lite.cpp
  src/Omega_h_yaml.cpp
)
set_source_files_properties( ${OMEGA_H_CUDA_FILES} 
   PROPERTIES 
   CUDA_SOURCE_PROPERTY_FORMAT OBJ 
) 

# Find lgr files
file(GLOB LGR_CUDA_FILES
    src/lgr_scope.cpp
    src/lgr_condition.cpp
    src/lgr_input_variables.cpp
    src/lgr_disc.cpp
    src/lgr_field.cpp
    src/lgr_fields.cpp
    src/lgr_hydro.cpp
    src/lgr_hyper_ep.cpp
    src/lgr_ideal_gas.cpp
    src/lgr_mie_gruneisen.cpp
    src/lgr_model.cpp
    src/lgr_models.cpp
    src/lgr_simulation.cpp
    src/lgr_subset.cpp
    src/lgr_subsets.cpp
    src/lgr_support.cpp
    src/lgr_supports.cpp
    src/lgr_when.cpp
    src/lgr_run.cpp
    src/lgr_factories.cpp
    src/lgr_response.cpp
    src/lgr_responses.cpp
    src/lgr_vtk_output.cpp
    src/lgr_scalar.cpp
    src/lgr_scalars.cpp
    src/lgr_csv_hist.cpp
    src/lgr_cmdline_hist.cpp
    src/lgr_node_scalar.cpp
    src/lgr_comparison.cpp
    src/lgr_l2_error.cpp
    src/lgr_ray_trace.cpp
    src/lgr_artificial_viscosity.cpp
    src/lgr_adapt.cpp
    src/lgr_remap.cpp
    src/lgr_flood.cpp
    src/lgr_internal_energy.cpp
    src/lgr_deformation_gradient.cpp
    src/lgr_j2_plasticity.cpp
    src/lgr_neo_hookean.cpp
    src/lgr_stvenant_kirchhoff.cpp
    src/lgr_riemann.cpp
    src/lgr_osh_output.cpp
    src/lgr_quadratic.cpp
    src/lgr_linear_algebra.cpp
    src/lgr_joule_heating.cpp
    src/lgr_circuit.cpp
    src/lgr_traction.cpp
    src/lgr_nodal_pressure.cpp
    src/lgr_pressure.cpp
    src/lgr_setup.cpp
)
set_source_files_properties( ${LGR_CUDA_FILES} 
   PROPERTIES 
   CUDA_SOURCE_PROPERTY_FORMAT OBJ 
)

# Find main lgr file
file(GLOB RUN_CUDA_FILES
   src/lgr.cpp
)
set_source_files_properties( ${RUN_CUDA_FILES} 
   PROPERTIES 
   CUDA_SOURCE_PROPERTY_FORMAT OBJ 
)

# Set properties before any compile commands are done
set(CUDA_SEPARABLE_COMPILATION OFF)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "-Domega_h_EXPORTS")
list(APPEND CUDA_NVCC_FLAGS "-Dlgr_library_EXPORTS")
list(APPEND CUDA_NVCC_FLAGS "-DLGR_DISABLE_APREPRO")
include_directories(src)
cuda_include_directories(src)

# Compile omega_h library
CUDA_COMPILE(OMEGA_H_O ${OMEGA_H_CUDA_FILES})
cuda_add_library(omega_h_lib "${OMEGA_H_O}")
set_property(TARGET omega_h_lib PROPERTY CUDA_STANDARD 11)

# Compile lgr library
CUDA_COMPILE(LGR_O ${LGR_CUDA_FILES})
cuda_add_library(lgr_lib "${LGR_O}")
set_property(TARGET lgr_lib PROPERTY CUDA_STANDARD 11)

# Compile executable
cuda_add_executable(lgr "${RUN_CUDA_FILES}")
target_link_libraries(lgr omega_h_lib)
target_link_libraries(lgr lgr_lib)
set_property(TARGET lgr PROPERTY CUDA_STANDARD 11)