#include <lgr_subsets.hpp>
#include <lgr_supports.hpp>

namespace lgr {

Subset* const& SubsetOfSupportPtr::operator()(Support* s) { return s->subset; }

Supports::Supports(Subsets& subsets_in) : subsets(subsets_in) {}

Support* Supports::get_support(
    EntityType entity_type, bool on_points, ClassNames const& class_names) {
  auto subset = subsets.get_subset(entity_type, class_names);
  auto out = get_support(subset, on_points);
  OMEGA_H_CHECK(out->subset->entity_type == entity_type);
  return out;
}

Support* Supports::get_support(Subset* subset, bool on_points) {
  auto& disc = subsets.disc;
  auto& by_subset = supports[on_points ? 1 : 0];
  auto it = by_subset.lower_bound(subset);
  if (it == by_subset.end() || (*it)->subset != subset) {
    Support* ptr;
    if (on_points)
      ptr = this->point_support_factory(disc, subset);
    else
      ptr = entity_support_factory(disc, subset);
    std::unique_ptr<Support> unique_ptr(ptr);
    storage.push_back(std::move(unique_ptr));
    it = by_subset.insert(it, ptr);
    OMEGA_H_CHECK(it != by_subset.end());
    OMEGA_H_CHECK(*it == ptr);
  }
  auto out = *it;
  OMEGA_H_CHECK(out->subset == subset);
  return out;
}

template <class Elem>
void Supports::set_elem() {
  this->point_support_factory = lgr::point_support_factory<Elem>;
}

#define LGR_EXPL_INST(Elem) template void Supports::set_elem<Elem>();
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr
