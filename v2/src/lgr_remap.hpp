#ifndef LGR_remap_HPP
#define LGR_remap_HPP

#include <lgr_element_types.hpp>
#include <lgr_remap_type.hpp>
#include <lgr_field_index.hpp>
#include <Omega_h_adapt.hpp>

namespace lgr {

struct Simulation;

struct RemapBase : public Omega_h::UserTransfer {
  Simulation& sim;
  std::map<RemapType, std::vector<std::string>> fields_to_remap;
  std::vector<FieldIndex> field_indices_to_remap;
  RemapBase(Simulation& sim_in);
  virtual void out_of_line_virtual_method();
  virtual void before_adapt() = 0;
  virtual void after_adapt() = 0;
};

template <class Elem>
RemapBase* remap_factory(Simulation& sim);

#define LGR_EXPL_INST(Elem) \
extern template RemapBase* remap_factory<Elem>(Simulation&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}

#endif
