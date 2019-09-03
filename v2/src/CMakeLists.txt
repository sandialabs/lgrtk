bob_config_header("${CMAKE_CURRENT_BINARY_DIR}/lgr_config.hpp")

add_library(lgr_library
    lgr_scope.cpp
    lgr_condition.cpp
    lgr_input_variables.cpp
    lgr_disc.cpp
    lgr_field.cpp
    lgr_fields.cpp
    lgr_hydro.cpp
    lgr_hyper_ep.cpp
    lgr_ideal_gas.cpp
    lgr_mie_gruneisen.cpp
    lgr_model.cpp
    lgr_models.cpp
    lgr_simulation.cpp
    lgr_subset.cpp
    lgr_subsets.cpp
    lgr_support.cpp
    lgr_supports.cpp
    lgr_when.cpp
    lgr_run.cpp
    lgr_factories.cpp
    lgr_response.cpp
    lgr_responses.cpp
    lgr_vtk_output.cpp
    lgr_scalar.cpp
    lgr_scalars.cpp
    lgr_csv_hist.cpp
    lgr_cmdline_hist.cpp
    lgr_node_scalar.cpp
    lgr_comparison.cpp
    lgr_l2_error.cpp
    lgr_ray_trace.cpp
    lgr_artificial_viscosity.cpp
    lgr_adapt.cpp
    lgr_remap.cpp
    lgr_flood.cpp
    lgr_internal_energy.cpp
    lgr_deformation_gradient.cpp
    lgr_j2_plasticity.cpp
    lgr_neo_hookean.cpp
    lgr_stvenant_kirchhoff.cpp
    lgr_riemann.cpp
    lgr_osh_output.cpp
    lgr_quadratic.cpp
    lgr_linear_algebra.cpp
    lgr_joule_heating.cpp
    lgr_circuit.cpp
    lgr_traction.cpp
    lgr_nodal_pressure.cpp
    lgr_pressure.cpp
    lgr_setup.cpp
    )

set_target_properties(lgr_library PROPERTIES OUTPUT_NAME lgr)

bob_library_includes(lgr_library)
bob_link_dependency(lgr_library PUBLIC Omega_h)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/lgr_config.hpp
    lgr_math.hpp
    lgr_element_types.hpp
    lgr_factories.hpp
    lgr_input_variables.hpp
    lgr_run.hpp
    lgr_for.hpp
    lgr_model.hpp
    lgr_field_index.hpp
    lgr_field_access.hpp
    lgr_remap_type.hpp
    lgr_class_names.hpp
    lgr_mapping.hpp
    lgr_disc.hpp
    lgr_entity_type.hpp
    lgr_subsets.hpp
    lgr_subset.hpp
    lgr_supports.hpp
    lgr_support.hpp
    lgr_field.hpp
    lgr_fields.hpp
    lgr_models.hpp
    lgr_scalar.hpp
    lgr_scalars.hpp
    lgr_response.hpp
    lgr_responses.hpp
    lgr_adapt.hpp
    lgr_remap.hpp
    lgr_simulation.hpp
    lgr_condition.hpp
    lgr_when.hpp
    lgr_flood.hpp
    lgr_element_functions.hpp
    lgr_setup.hpp
    lgr_circuit.hpp
    lgr_globals.hpp
    lgr_linear_algebra.hpp
    DESTINATION include)

if(LGR_COMPTET)
  install(FILES
    lgr_comptet_functions.hpp
    DESTINATION include)
endif()

add_executable(lgr_executable lgr.cpp)
target_link_libraries(lgr_executable lgr_library)
set_target_properties(lgr_executable PROPERTIES
    OUTPUT_NAME lgr
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

bob_export_target(lgr_library)
bob_export_target(lgr_executable)

bob_end_subdir()
