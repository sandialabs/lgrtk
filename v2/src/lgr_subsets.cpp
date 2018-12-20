#include <Omega_h_map.hpp>
#include <Omega_h_profile.hpp>
#include <lgr_disc.hpp>
#include <lgr_subset.hpp>
#include <lgr_subsets.hpp>

namespace lgr {

ClassNames const& ClassNamesOfSubset::operator()(Subset const& ss) const {
  return ss.class_names;
}

SubsetPair const& SubsetPairOfSubsetBridge::operator()(
    SubsetBridge const& ssb) {
  return ssb.subsets;
}

void TypeSubsets::forget_disc() {
  for (auto& b : bridges) {
    b.forget_disc();
  }
  for (auto& s : by_class_names) {
    s.forget_disc();
  }
}

void TypeSubsets::learn_disc(Subsets& subsets) {
  for (auto& s : by_class_names) {
    s.learn_disc();
  }
  for (auto& b : bridges) {
    b.learn_disc(subsets, *this);
  }
}

Subsets::Subsets(Disc& disc_in) : disc(disc_in) {}

Subset* Subsets::get_subset(EntityType type, ClassNames const& class_names) {
  auto& of_type = by_type[type];
  auto& by_class_names = of_type.by_class_names;
  auto it = by_class_names.lower_bound(class_names);
  if (it == by_class_names.end() || it->class_names != class_names) {
    Subset subset(disc, type, class_names);
    it = by_class_names.insert(it, std::move(subset));
  }
  auto out = &(*it);
  OMEGA_H_CHECK(out->entity_type == type);
  OMEGA_H_CHECK(out->class_names == class_names);
  return out;
}

void SubsetBridge::forget_disc() {
  mapping.is_identity = false;
  mapping.things = decltype(mapping.things)();
}

void SubsetBridge::learn_disc(Subsets& all_subsets, TypeSubsets& type_subsets) {
  auto from = this->subsets.first;
  auto to = this->subsets.second;
  if (from == to) {
    mapping.is_identity = true;
  } else if (from->is_identity() && to->is_identity()) {
    mapping.is_identity = true;
  } else if (to->is_identity()) {
    mapping = from->mapping;
  } else if (from->is_identity()) {
    Omega_h_fail(
        "bridging from identity into "
        "a subset not allowed\n");
  } else {
    auto& by_class_names = type_subsets.by_class_names;
    auto it2 = by_class_names.find(all_subsets.disc.covering_class_names());
    OMEGA_H_CHECK(it2 != by_class_names.end());
    auto& identity = *it2;
    auto ntotal = identity.count();
    auto to_map = to->mapping.things;
    auto inv_to_map = all_subsets.acquire_inverse(to_map, ntotal);
    mapping.things = Omega_h::unmap(from->mapping.things, inv_to_map, 1);
    all_subsets.release_inverse(to_map);
  }
}

SubsetBridge* Subsets::get_bridge(Subset* from, Subset* to) {
  if (!(from->entity_type == to->entity_type)) {
    Omega_h_fail("entity type %d != %d\n", from->entity_type, to->entity_type);
  }
  OMEGA_H_CHECK(from->entity_type == to->entity_type);
  auto& of_type = by_type[from->entity_type];
  auto& bridges = of_type.bridges;
  auto key = std::make_pair(from, to);
  auto it = bridges.lower_bound(key);
  if (it == bridges.end() || it->subsets != key) {
    SubsetBridge b;
    b.subsets = key;
    b.learn_disc(*this, of_type);
    it = bridges.insert(it, b);
  }
  return &(*it);
}

void Subsets::forget_disc() {
  Omega_h::ScopedTimer timer("Subsets::forget_disc");
  for (int i = 0; i < 4; ++i) {
    by_type[i].forget_disc();
  }
}

void Subsets::learn_disc() {
  Omega_h::ScopedTimer timer("Subsets::learn_disc");
  for (int i = 0; i < 4; ++i) {
    by_type[i].learn_disc(*this);
  }
}

Omega_h::LOs Subsets::acquire_inverse(Omega_h::LOs a2b, int nb) {
  if (!inverse_buffer.exists() || inverse_buffer.size() < nb) {
    inverse_buffer = decltype(inverse_buffer)(nb, -1);
  }
  Omega_h::inject_map(a2b, inverse_buffer);
  return inverse_buffer;
}

void Subsets::release_inverse(Omega_h::LOs a2b) {
  Omega_h::map_value_into(-1, a2b, inverse_buffer);
}

}  // namespace lgr
