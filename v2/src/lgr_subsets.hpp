#ifndef LGR_SUBSETS_HPP
#define LGR_SUBSETS_HPP

#include <Omega_h_rbtree.hpp>
#include <lgr_subset.hpp>
#include <memory>
#include <vector>

namespace lgr {

struct ClassNamesOfSubset {
  ClassNames const& operator()(Subset const& ss) const;
};

using SubsetPair = std::pair<Subset*, Subset*>;

struct TypeSubsets;
struct Subsets;

struct SubsetBridge {
  SubsetPair subsets;
  Mapping mapping;
  void forget_disc();
  void learn_disc(Subsets&, TypeSubsets&);
};

struct SubsetPairOfSubsetBridge {
  SubsetPair const& operator()(SubsetBridge const& ssb);
};

struct TypeSubsets {
  Omega_h::rb_tree<ClassNames, Subset, ClassNamesOfSubset> by_class_names;
  Omega_h::rb_tree<SubsetPair, SubsetBridge, SubsetPairOfSubsetBridge> bridges;
  void forget_disc();
  void learn_disc(Subsets&);
};

struct Subsets {
  Disc& disc;
  TypeSubsets by_type[4];
  Subsets(Disc& disc_in);
  Subset* get_subset(EntityType type, ClassNames const& class_names);
  SubsetBridge* get_bridge(Subset* from, Subset* to);
  void forget_disc();
  void learn_disc();
  Omega_h::LOs acquire_inverse(Omega_h::LOs a2b, int nb);
  void release_inverse(Omega_h::LOs a2b);
  Omega_h::Write<int> inverse_buffer;
};

}  // namespace lgr

#endif
