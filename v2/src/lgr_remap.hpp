#ifndef LGR_remap_HPP
#define LGR_remap_HPP

#include <Omega_h_adapt.hpp>
#include <Omega_h_input.hpp>
#include <lgr_element_types.hpp>
#include <lgr_field_index.hpp>
#include <lgr_remap_type.hpp>

namespace lgr {

struct Simulation;

struct RemapBase : public Omega_h::UserTransfer {
  Simulation& sim;
  std::map<RemapType, std::vector<std::string>> fields_to_remap;
  std::vector<FieldIndex> field_indices_to_remap;
  double axis_angle_tolerance;
  RemapBase(Simulation& sim_in, Omega_h::InputMap& pl);
  virtual void out_of_line_virtual_method();
  virtual void before_adapt() = 0;
  virtual void after_adapt() = 0;
};

template <class Elem>
RemapBase* remap_factory(Simulation& sim, Omega_h::InputMap& pl);

#define LGR_EXPL_INST(Elem)                                                    \
  extern template RemapBase* remap_factory<Elem>(                              \
      Simulation&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

Omega_h::Write<double> allocate_and_fill_with_same(Omega_h::Mesh& new_mesh,
    int ent_dim, int ncomps, Omega_h::LOs same_ents2old_ents,
    Omega_h::LOs same_ents2new_ents, Omega_h::Reals old_data);

}  // namespace lgr

#endif
