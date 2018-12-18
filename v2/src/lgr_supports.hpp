#ifndef LGR_SUPPORTS_HPP
#define LGR_SUPPORTS_HPP

#include <Omega_h_rbtree.hpp>
#include <lgr_class_names.hpp>
#include <lgr_entity_type.hpp>
#include <lgr_support.hpp>
#include <memory>
#include <vector>

namespace lgr {

struct Subset;
struct Subsets;

struct SubsetOfSupportPtr {
  Subset* const& operator()(Support* s);
};

struct Supports {
  Subsets& subsets;
  std::vector<std::unique_ptr<Support>> storage;
  using BySubset = Omega_h::rb_tree<Subset*, Support*, SubsetOfSupportPtr>;
  using SupportPtr = Support*;
  using PointSupportFactory = std::function<SupportPtr(Disc&, Subset*)>;
  BySubset supports[2];
  PointSupportFactory point_support_factory;
  Supports(Subsets& subsets);
  Support* get_support(
      EntityType entity_type, bool on_points, ClassNames const& class_names);
  Support* get_support(Subset* subset, bool on_points);
  template <class Elem>
  void set_elem();
};

#define LGR_EXPL_INST(Elem) extern template void Supports::set_elem<Elem>();
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr

#endif
