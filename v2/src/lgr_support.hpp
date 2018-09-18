#ifndef LGR_SUPPORT_HPP
#define LGR_SUPPORT_HPP

#include <lgr_element_types.hpp>

namespace lgr {

struct Subset;
struct Disc;

struct Support {
  Disc& disc;
  Subset* subset;
  Omega_h::Read<double> cached_coords;
  double cached_time;
  Support(Disc& disc_in, Subset* subset_in);
  virtual ~Support() = default;
  Omega_h::Read<double> ask_coords(
      double time,
      Omega_h::Read<double> node_coords);
  virtual void out_of_line_virtual_method();
  virtual int count() = 0;
  virtual Omega_h::Read<double> interpolate_nodal(
      int ncomps,
      Omega_h::Read<double> node_coords) = 0;
  virtual bool on_points() = 0;
};

Support* entity_support_factory(Disc& disc, Subset* subset);

template <class Elem>
Support* point_support_factory(Disc& disc, Subset* subset);

#define LGR_EXPL_INST(Elem) \
extern template Support* point_support_factory<Elem>(Disc& disc, Subset*);
LGR_EXPL_INST_ELEMS_AND_SIDES
#undef LGR_EXPL_INST

}

#endif

