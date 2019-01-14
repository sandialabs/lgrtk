#include <Omega_h_profile.hpp>
#include <lgr_element_functions.hpp>
#include <lgr_for.hpp>
#include <lgr_remap.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

RemapBase::RemapBase(Simulation& sim_in) : sim(sim_in) {
  for (auto& field_ptr : sim.fields.storage) {
    if (field_ptr->remap_type != RemapType::NONE) {
      fields_to_remap[field_ptr->remap_type].push_back(field_ptr->long_name);
      field_indices_to_remap.push_back(sim.fields.find(field_ptr->long_name));
      if (field_ptr->remap_type == RemapType::POSITIVE_DETERMINANT) {
        fields_to_remap[RemapType::PER_UNIT_VOLUME].push_back(
            field_ptr->long_name);
      }
    }
  }
}

void RemapBase::out_of_line_virtual_method() {}

struct VolumeWeighter {
  Omega_h::Reals points_to_w;
  VolumeWeighter(Omega_h::Mesh& mesh) {
    points_to_w = mesh.get_array<double>(mesh.dim(), "weight");
  }
  OMEGA_H_DEVICE double get_weight(int point) const {
    return points_to_w[point];
  }
};

struct MassWeighter {
  Omega_h::Reals points_to_w;
  Omega_h::Reals points_to_rho;
  MassWeighter(Omega_h::Mesh& mesh) {
    points_to_w = mesh.get_array<double>(mesh.dim(), "weight");
    points_to_rho = mesh.get_array<double>(mesh.dim(), "density");
  }
  OMEGA_H_DEVICE double get_weight(int point) const {
    return points_to_w[point] * points_to_rho[point];
  }
};

static void remap_old_class_id(Omega_h::Mesh& old_mesh, Omega_h::Mesh& new_mesh,
    Omega_h::LOs prods2new_ents, Omega_h::LOs same_ents2old_ents,
    Omega_h::LOs same_ents2new_ents) {
  if (!old_mesh.has_tag(old_mesh.dim(), "old class_id")) return;
  OMEGA_H_TIME_FUNCTION;
  auto old_data =
      old_mesh.get_array<Omega_h::ClassId>(old_mesh.dim(), "old class_id");
  auto new_data = Omega_h::Write<Omega_h::ClassId>(new_mesh.nelems());
  auto same_functor = OMEGA_H_LAMBDA(int same_elem) {
    auto old_elem = same_ents2old_ents[same_elem];
    auto new_elem = same_ents2new_ents[same_elem];
    new_data[new_elem] = old_data[old_elem];
  };
  parallel_for(same_ents2old_ents.size(), std::move(same_functor));
  auto class_ids =
      new_mesh.get_array<Omega_h::ClassId>(new_mesh.dim(), "class_id");
  auto prod_functor = OMEGA_H_LAMBDA(int prod_elem) {
    auto elem = prods2new_ents[prod_elem];
    new_data[elem] = class_ids[elem];
  };
  parallel_for(prods2new_ents.size(), std::move(prod_functor));
  new_mesh.add_tag(new_mesh.dim(), "old class_id", 1, read(new_data));
}

template <class Elem>
struct Remap : public RemapBase {
  Remap(Simulation& sim_in) : RemapBase(sim_in) {}
  void before_adapt() override final;
  Omega_h::Write<double> allocate_and_fill_with_same(Omega_h::Mesh& new_mesh,
      int ent_dim, int ncomps, Omega_h::LOs same_ents2old_ents,
      Omega_h::LOs same_ents2new_ents, Omega_h::Reals old_data);
  Omega_h::Write<double> setup_new_shape_data(Omega_h::Mesh& old_mesh,
      Omega_h::Mesh& new_mesh, Omega_h::LOs same_ents2old_ents,
      Omega_h::LOs same_ents2new_ents, std::string const& name);
  void remap_shape(Omega_h::Mesh& old_mesh, Omega_h::Mesh& new_mesh,
      Omega_h::LOs keys2prods, Omega_h::LOs prods2new_ents,
      Omega_h::LOs same_ents2old_ents, Omega_h::LOs same_ents2new_ents);
  template <class Weighter, int ncomps>
  void refine_point_remap_ncomps(Omega_h::Mesh& old_mesh,
      Omega_h::Mesh& new_mesh, int key_dim, int prod_dim, Omega_h::LOs keys2kds,
      Omega_h::LOs keys2prods, Omega_h::LOs prods2new_ents,
      Omega_h::LOs same_ents2old_ents, Omega_h::LOs same_ents2new_ents,
      Omega_h::Tag<double> const* tag);
  void coarsen_point_remap(Omega_h::Mesh& old_mesh, Omega_h::Mesh& new_mesh,
      int prod_dim, Omega_h::LOs keys2prods, Omega_h::LOs key_doms2doms,
      Omega_h::LOs prods2new_ents, Omega_h::LOs same_ents2old_ents,
      Omega_h::LOs same_ents2new_ents, std::string const& name);
  template <class Weighter, int ncomps>
  void swap_point_remap_ncomps(Omega_h::Mesh& old_mesh, Omega_h::Mesh& new_mesh,
      int key_dim, int prod_dim, Omega_h::LOs keys2kds, Omega_h::LOs keys2prods,
      Omega_h::LOs prods2new_ents, Omega_h::LOs same_ents2old_ents,
      Omega_h::LOs same_ents2new_ents, Omega_h::Tag<double> const* tag);
  template <class Weighter>
  void refine_point_remap(Omega_h::Mesh& old_mesh, Omega_h::Mesh& new_mesh,
      int key_dim, int prod_dim, Omega_h::LOs keys2kds, Omega_h::LOs keys2prods,
      Omega_h::LOs prods2new_ents, Omega_h::LOs same_ents2old_ents,
      Omega_h::LOs same_ents2new_ents, std::string const& name);
  template <class Weighter>
  void swap_point_remap(Omega_h::Mesh& old_mesh, Omega_h::Mesh& new_mesh,
      int key_dim, int prod_dim, Omega_h::LOs keys2kds, Omega_h::LOs keys2prods,
      Omega_h::LOs prods2new_ents, Omega_h::LOs same_ents2old_ents,
      Omega_h::LOs same_ents2new_ents, std::string const& name);
  void refine(Omega_h::Mesh& old_mesh, Omega_h::Mesh& new_mesh,
      Omega_h::LOs keys2edges, Omega_h::LOs keys2midverts, int prod_dim,
      Omega_h::LOs keys2prods, Omega_h::LOs prods2new_ents,
      Omega_h::LOs same_ents2old_ents,
      Omega_h::LOs same_ents2new_ents) override final;
  void coarsen(Omega_h::Mesh& old_mesh, Omega_h::Mesh& new_mesh,
      Omega_h::LOs /*keys2verts*/, Omega_h::Adj keys2doms, int prod_dim,
      Omega_h::LOs prods2new_ents, Omega_h::LOs same_ents2old_ents,
      Omega_h::LOs same_ents2new_ents) override final;
  void swap_copy_verts(
      Omega_h::Mesh& old_mesh, Omega_h::Mesh& new_mesh) override final;
  void swap(Omega_h::Mesh& old_mesh, Omega_h::Mesh& new_mesh, int prod_dim,
      Omega_h::LOs keys2edges, Omega_h::LOs keys2prods,
      Omega_h::LOs prods2new_ents, Omega_h::LOs same_ents2old_ents,
      Omega_h::LOs same_ents2new_ents) override final;
  void after_adapt() override final;
};

template <class Elem>
void Remap<Elem>::before_adapt() {
  Omega_h::ScopedTimer timer("Remap::before_adapt");
  for (auto& name : fields_to_remap[RemapType::POSITIVE_DETERMINANT]) {
    auto const fi = sim.fields.find(name);
    auto const points_to_F = sim.getset(fi);
    auto const npoints = sim.fields[fi].support->count();
    auto functor = OMEGA_H_LAMBDA(int const point) {
      auto const F = getfull<Elem>(points_to_F, point);
      OMEGA_H_CHECK(determinant(F) > 0.0);
      auto const log_F = Omega_h::log_polar(F);
      setfull<Elem>(points_to_F, point, log_F);
    };
    parallel_for(npoints, std::move(functor));
  }
  sim.fields.copy_to_omega_h(sim.disc, field_indices_to_remap);
}

template <class Elem>
Omega_h::Write<double> Remap<Elem>::allocate_and_fill_with_same(Omega_h::Mesh& new_mesh,
    int ent_dim, int ncomps, Omega_h::LOs same_ents2old_ents,
    Omega_h::LOs same_ents2new_ents, Omega_h::Reals old_data) {
  auto const new_data = Omega_h::Write<double>(new_mesh.nents(ent_dim) * ncomps);
  auto same_functor = OMEGA_H_LAMBDA(int same_ent) {
    auto old_ent = same_ents2old_ents[same_ent];
    auto new_ent = same_ents2new_ents[same_ent];
    for (int comp = 0; comp < ncomps; ++comp) {
      new_data[new_ent * ncomps + comp] = old_data[old_ent * ncomps + comp];
    }
  };
  parallel_for(same_ents2old_ents.size(), std::move(same_functor));
  return new_data;
}

template <class Elem>
Omega_h::Write<double> Remap<Elem>::setup_new_shape_data(Omega_h::Mesh& old_mesh,
    Omega_h::Mesh& new_mesh, Omega_h::LOs same_ents2old_ents,
    Omega_h::LOs same_ents2new_ents, std::string const& name) {
  auto tag = old_mesh.get_tag<double>(old_mesh.dim(), name);
  auto ncomps = tag->ncomps();
  auto old_data = tag->array();
  return allocate_and_fill_with_same(new_mesh, new_mesh.dim(), ncomps,
      same_ents2old_ents, same_ents2new_ents, old_data);
}

template <class Elem>
void Remap<Elem>::remap_shape(Omega_h::Mesh& old_mesh, Omega_h::Mesh& new_mesh,
    Omega_h::LOs keys2prods, Omega_h::LOs prods2new_ents,
    Omega_h::LOs same_ents2old_ents, Omega_h::LOs same_ents2new_ents) {
  auto new_weights = setup_new_shape_data(
      old_mesh, new_mesh, same_ents2old_ents, same_ents2new_ents, "weight");
  auto new_gradients = setup_new_shape_data(
      old_mesh, new_mesh, same_ents2old_ents, same_ents2new_ents, "gradient");
  auto new_dt_h = setup_new_shape_data(old_mesh, new_mesh, same_ents2old_ents,
      same_ents2new_ents, "time step length");
  auto new_visc_h = setup_new_shape_data(old_mesh, new_mesh,
      same_ents2old_ents, same_ents2new_ents, "viscosity length");
  auto new_coords = new_mesh.coords();
  auto new_elems2verts = new_mesh.ask_elem_verts();
  auto new_functor = OMEGA_H_LAMBDA(int key) {
    for (auto prod = keys2prods[key]; prod < keys2prods[key + 1]; ++prod) {
      auto new_elem = prods2new_ents[prod];
      auto elem_verts =
          Omega_h::gather_verts<Elem::nodes>(new_elems2verts, new_elem);
      auto elem_node_coords = Omega_h::gather_vectors<Elem::nodes, Elem::dim>(
          new_coords, elem_verts);
      auto shape = Elem::shape(elem_node_coords);
      new_dt_h[new_elem] = shape.lengths.time_step_length;
      new_visc_h[new_elem] = shape.lengths.viscosity_length;
      for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
        auto pt = new_elem * Elem::points + elem_pt;
        setgrads<Elem>(new_gradients, pt, shape.basis_gradients[elem_pt]);
        new_weights[pt] = shape.weights[elem_pt];
      }
    }
  };
  parallel_for(keys2prods.size() - 1, std::move(new_functor));
  new_mesh.add_tag(
      new_mesh.dim(), "weight", Elem::points, Omega_h::read(new_weights));
  new_mesh.add_tag(new_mesh.dim(), "gradient",
      Elem::points * Elem::nodes * Elem::dim, Omega_h::read(new_gradients));
  new_mesh.add_tag(
      new_mesh.dim(), "time step length", 1, Omega_h::read(new_dt_h));
  new_mesh.add_tag(
      new_mesh.dim(), "viscosity length", 1, Omega_h::read(new_visc_h));
}

template <class Elem>
template <class Weighter, int ncomps>
void Remap<Elem>::refine_point_remap_ncomps(Omega_h::Mesh& old_mesh,
    Omega_h::Mesh& new_mesh, int key_dim, int prod_dim, Omega_h::LOs keys2kds,
    Omega_h::LOs keys2prods, Omega_h::LOs prods2new_ents,
    Omega_h::LOs same_ents2old_ents, Omega_h::LOs same_ents2new_ents,
    Omega_h::Tag<double> const* tag) {
  auto old_data = tag->array();
  auto new_data = allocate_and_fill_with_same(new_mesh, new_mesh.dim(),
      tag->ncomps(), same_ents2old_ents, same_ents2new_ents, old_data);
  auto kds2doms = old_mesh.ask_graph(key_dim, prod_dim);
  Weighter weighter(old_mesh);
  auto new_functor = OMEGA_H_LAMBDA(int key) {
    auto kd = keys2kds[key];
    auto prod = keys2prods[key];
    auto const begin = kds2doms.a2ab[kd];
    auto const end = kds2doms.a2ab[kd + 1];
    for (auto kd_dom = begin; kd_dom < end; ++kd_dom) {
      auto dom = kds2doms.ab2b[kd_dom];
      auto value = zero_vector<ncomps>();
      auto weight_sum = 0.0;
      for (int dom_pt = 0; dom_pt < Elem::points; ++dom_pt) {
        auto old_point = dom * Elem::points + dom_pt;
        auto old_weight = weighter.get_weight(old_point);
        weight_sum += old_weight;
        for (int comp = 0; comp < ncomps; ++comp) {
          value[comp] += old_weight * old_data[old_point * ncomps + comp];
        }
      }
      for (int comp = 0; comp < ncomps; ++comp) {
        value[comp] /= weight_sum;
      }
      for (int child = 0; child < 2; ++child) {
        auto new_elem = prods2new_ents[prod];
        for (int child_pt = 0; child_pt < Elem::points; ++child_pt) {
          auto new_point = new_elem * Elem::points + child_pt;
          for (int comp = 0; comp < ncomps; ++comp) {
            new_data[new_point * ncomps + comp] = value[comp];
          }
        }
        ++prod;
      }
    }
  };
  parallel_for(keys2kds.size(), std::move(new_functor));
  new_mesh.add_tag(
      new_mesh.dim(), tag->name(), tag->ncomps(), Omega_h::read(new_data));
}

template <class Elem>
void Remap<Elem>::coarsen_point_remap(Omega_h::Mesh& old_mesh, Omega_h::Mesh& new_mesh,
    int prod_dim, Omega_h::LOs keys2prods, Omega_h::LOs key_doms2doms,
    Omega_h::LOs prods2new_ents, Omega_h::LOs same_ents2old_ents,
    Omega_h::LOs same_ents2new_ents, std::string const& name) {
  auto tag = old_mesh.get_tag<double>(prod_dim, name);
  auto old_data = tag->array();
  auto ncomps = divide_no_remainder(tag->ncomps(), Elem::points);
  auto new_data = allocate_and_fill_with_same(new_mesh, new_mesh.dim(),
      tag->ncomps(), same_ents2old_ents, same_ents2new_ents, old_data);
  auto new_functor = OMEGA_H_LAMBDA(int key) {
    for (auto prod = keys2prods[key]; prod < keys2prods[key + 1]; ++prod) {
      auto const key_dom = prod;
      auto const old_elem = key_doms2doms[key_dom];
      auto const new_elem = prods2new_ents[prod];
      for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
        auto const old_pt = old_elem * Elem::points + elem_pt;
        auto const new_pt = new_elem * Elem::points + elem_pt;
        for (int comp = 0; comp < ncomps; ++comp) {
          new_data[new_pt * ncomps + comp] = old_data[old_pt * ncomps + comp];
        }
      }
    }
  };
  parallel_for(keys2prods.size() - 1, std::move(new_functor));
  new_mesh.add_tag(
      new_mesh.dim(), tag->name(), tag->ncomps(), Omega_h::read(new_data));
}

template <class Elem>
template <class Weighter, int ncomps>
void Remap<Elem>::swap_point_remap_ncomps(Omega_h::Mesh& old_mesh, Omega_h::Mesh& new_mesh,
    int key_dim, int prod_dim, Omega_h::LOs keys2kds, Omega_h::LOs keys2prods,
    Omega_h::LOs prods2new_ents, Omega_h::LOs same_ents2old_ents,
    Omega_h::LOs same_ents2new_ents, Omega_h::Tag<double> const* tag) {
  auto old_data = tag->array();
  auto new_data = allocate_and_fill_with_same(new_mesh, new_mesh.dim(),
      tag->ncomps(), same_ents2old_ents, same_ents2new_ents, old_data);
  auto kds2doms = old_mesh.ask_graph(key_dim, prod_dim);
  Weighter weighter(old_mesh);
  auto new_functor = OMEGA_H_LAMBDA(int key) {
    auto kd = keys2kds[key];
    auto value = zero_vector<ncomps>();
    auto weight_sum = 0.0;
    auto const begin = kds2doms.a2ab[kd];
    auto const end = kds2doms.a2ab[kd + 1];
    for (auto kd_dom = begin; kd_dom < end; ++kd_dom) {
      auto dom = kds2doms.ab2b[kd_dom];
      for (int dom_pt = 0; dom_pt < Elem::points; ++dom_pt) {
        auto old_point = dom * Elem::points + dom_pt;
        auto old_weight = weighter.get_weight(old_point);
        weight_sum += old_weight;
        for (int comp = 0; comp < ncomps; ++comp) {
          value[comp] += old_weight * old_data[old_point * ncomps + comp];
        }
      }
    }
    for (int comp = 0; comp < ncomps; ++comp) {
      value[comp] /= weight_sum;
    }
    for (auto prod = keys2prods[key]; prod < keys2prods[key + 1]; ++prod) {
      auto new_elem = prods2new_ents[prod];
      for (int prod_pt = 0; prod_pt < Elem::points; ++prod_pt) {
        auto new_point = new_elem * Elem::points + prod_pt;
        for (int comp = 0; comp < ncomps; ++comp) {
          new_data[new_point * ncomps + comp] = value[comp];
        }
      }
    }
  };
  parallel_for(keys2kds.size(), std::move(new_functor));
  new_mesh.add_tag(
      new_mesh.dim(), tag->name(), tag->ncomps(), Omega_h::read(new_data));
}

template <class Elem>
template <class Weighter>
void Remap<Elem>::refine_point_remap(Omega_h::Mesh& old_mesh, Omega_h::Mesh& new_mesh,
    int key_dim, int prod_dim, Omega_h::LOs keys2kds, Omega_h::LOs keys2prods,
    Omega_h::LOs prods2new_ents, Omega_h::LOs same_ents2old_ents,
    Omega_h::LOs same_ents2new_ents, std::string const& name) {
  auto tag = old_mesh.get_tag<double>(prod_dim, name);
  auto ncomps = divide_no_remainder(tag->ncomps(), Elem::points);
  if (ncomps == 1) {
    refine_point_remap_ncomps<Weighter, 1>(old_mesh, new_mesh, key_dim,
        prod_dim, keys2kds, keys2prods, prods2new_ents, same_ents2old_ents,
        same_ents2new_ents, tag);
  } else if (ncomps == Omega_h::symm_ncomps(Elem::dim)) {
    refine_point_remap_ncomps<Weighter, Omega_h::symm_ncomps(Elem::dim)>(
        old_mesh, new_mesh, key_dim, prod_dim, keys2kds, keys2prods,
        prods2new_ents, same_ents2old_ents, same_ents2new_ents, tag);
  } else if (ncomps == Omega_h::square(Elem::dim)) {
    refine_point_remap_ncomps<Weighter, Omega_h::square(Elem::dim)>(old_mesh,
        new_mesh, key_dim, prod_dim, keys2kds, keys2prods, prods2new_ents,
        same_ents2old_ents, same_ents2new_ents, tag);
  } else {
    Omega_h_fail("unexpected refine point remap ncomps %d\n", ncomps);
  }
}

template <class Elem>
template <class Weighter>
void Remap<Elem>::swap_point_remap(Omega_h::Mesh& old_mesh, Omega_h::Mesh& new_mesh,
    int key_dim, int prod_dim, Omega_h::LOs keys2kds, Omega_h::LOs keys2prods,
    Omega_h::LOs prods2new_ents, Omega_h::LOs same_ents2old_ents,
    Omega_h::LOs same_ents2new_ents, std::string const& name) {
  auto tag = old_mesh.get_tag<double>(prod_dim, name);
  auto ncomps = divide_no_remainder(tag->ncomps(), Elem::points);
  if (ncomps == 1) {
    swap_point_remap_ncomps<Weighter, 1>(old_mesh, new_mesh, key_dim,
        prod_dim, keys2kds, keys2prods, prods2new_ents, same_ents2old_ents,
        same_ents2new_ents, tag);
  } else if (ncomps == Omega_h::symm_ncomps(Elem::dim)) {
    swap_point_remap_ncomps<Weighter, Omega_h::symm_ncomps(Elem::dim)>(
        old_mesh, new_mesh, key_dim, prod_dim, keys2kds, keys2prods,
        prods2new_ents, same_ents2old_ents, same_ents2new_ents, tag);
  } else if (ncomps == Omega_h::square(Elem::dim)) {
    swap_point_remap_ncomps<Weighter, Omega_h::square(Elem::dim)>(old_mesh,
        new_mesh, key_dim, prod_dim, keys2kds, keys2prods, prods2new_ents,
        same_ents2old_ents, same_ents2new_ents, tag);
  } else {
    Omega_h_fail("unexpected weighted ncomps %d\n", ncomps);
  }
}

template <class Elem>
void Remap<Elem>::refine(Omega_h::Mesh& old_mesh, Omega_h::Mesh& new_mesh,
    Omega_h::LOs keys2edges, Omega_h::LOs keys2midverts, int prod_dim,
    Omega_h::LOs keys2prods, Omega_h::LOs prods2new_ents,
    Omega_h::LOs same_ents2old_ents,
    Omega_h::LOs same_ents2new_ents) {
  if (prod_dim == 0) {
    for (auto& name : fields_to_remap[RemapType::NODAL]) {
      auto tag = old_mesh.get_tag<double>(0, name);
      auto ncomps = tag->ncomps();
      auto old_data = tag->array();
      auto new_data = allocate_and_fill_with_same(new_mesh, 0, ncomps,
          same_ents2old_ents, same_ents2new_ents, old_data);
      auto old_edges2verts = old_mesh.ask_verts_of(1);
      auto interp_functor = OMEGA_H_LAMBDA(int key) {
        auto new_vert = keys2midverts[key];
        auto old_edge = keys2edges[key];
        auto old_vert0 = old_edges2verts[old_edge * 2 + 0];
        auto old_vert1 = old_edges2verts[old_edge * 2 + 1];
        for (int comp = 0; comp < ncomps; ++comp) {
          new_data[new_vert * ncomps + comp] =
              (1.0 / 2.0) * (old_data[old_vert0 * ncomps + comp] +
                                old_data[old_vert1 * ncomps + comp]);
        }
      };
      parallel_for(keys2edges.size(), std::move(interp_functor));
      new_mesh.add_tag(0, name, ncomps, Omega_h::read(new_data));
    }
  }
  if (prod_dim == old_mesh.dim()) {
    remap_shape(old_mesh, new_mesh, keys2prods, prods2new_ents,
        same_ents2old_ents, same_ents2new_ents);
    for (auto& name : fields_to_remap[RemapType::PER_UNIT_VOLUME]) {
      refine_point_remap<VolumeWeighter>(old_mesh, new_mesh, 1, prod_dim,
          keys2edges, keys2prods, prods2new_ents, same_ents2old_ents,
          same_ents2new_ents, name);
    }
    for (auto& name : fields_to_remap[RemapType::PER_UNIT_MASS]) {
      refine_point_remap<MassWeighter>(old_mesh, new_mesh, 1, prod_dim,
          keys2edges, keys2prods, prods2new_ents, same_ents2old_ents,
          same_ents2new_ents, name);
    }
    remap_old_class_id(old_mesh, new_mesh, prods2new_ents, same_ents2old_ents,
        same_ents2new_ents);
  }
}

template <class Elem>
void Remap<Elem>::coarsen(Omega_h::Mesh& old_mesh, Omega_h::Mesh& new_mesh,
    Omega_h::LOs /*keys2verts*/, Omega_h::Adj keys2doms, int prod_dim,
    Omega_h::LOs prods2new_ents, Omega_h::LOs same_ents2old_ents,
    Omega_h::LOs same_ents2new_ents) {
  if (prod_dim == 0) {
    for (auto& name : fields_to_remap[RemapType::NODAL]) {
      auto tag = old_mesh.get_tag<double>(0, name);
      auto ncomps = tag->ncomps();
      auto old_data = tag->array();
      auto new_data = allocate_and_fill_with_same(new_mesh, 0, ncomps,
          same_ents2old_ents, same_ents2new_ents, old_data);
      new_mesh.add_tag(0, name, ncomps, Omega_h::read(new_data));
    }
  }
  if (prod_dim == old_mesh.dim()) {
    remap_shape(old_mesh, new_mesh, keys2doms.a2ab, prods2new_ents,
        same_ents2old_ents, same_ents2new_ents);
    for (auto& name : fields_to_remap[RemapType::PER_UNIT_VOLUME]) {
      coarsen_point_remap(old_mesh, new_mesh, prod_dim, keys2doms.a2ab,
          keys2doms.ab2b, prods2new_ents, same_ents2old_ents,
          same_ents2new_ents, name);
    }
    for (auto& name : fields_to_remap[RemapType::PER_UNIT_MASS]) {
      coarsen_point_remap(old_mesh, new_mesh, prod_dim, keys2doms.a2ab,
          keys2doms.ab2b, prods2new_ents, same_ents2old_ents,
          same_ents2new_ents, name);
    }
    remap_old_class_id(old_mesh, new_mesh, prods2new_ents, same_ents2old_ents,
        same_ents2new_ents);
  }
}

template <class Elem>
void Remap<Elem>::swap_copy_verts(
    Omega_h::Mesh& old_mesh, Omega_h::Mesh& new_mesh) {
  for (auto& name : fields_to_remap[RemapType::NODAL]) {
    auto tag = old_mesh.get_tag<double>(0, name);
    auto ncomps = tag->ncomps();
    auto old_data = tag->array();
    new_mesh.add_tag(0, name, ncomps, old_data);
  }
}

template <class Elem>
void Remap<Elem>::swap(Omega_h::Mesh& old_mesh, Omega_h::Mesh& new_mesh, int prod_dim,
    Omega_h::LOs keys2edges, Omega_h::LOs keys2prods,
    Omega_h::LOs prods2new_ents, Omega_h::LOs same_ents2old_ents,
    Omega_h::LOs same_ents2new_ents) {
  if (prod_dim == old_mesh.dim()) {
    remap_shape(old_mesh, new_mesh, keys2prods, prods2new_ents,
        same_ents2old_ents, same_ents2new_ents);
    for (auto& name : fields_to_remap[RemapType::PER_UNIT_VOLUME]) {
      swap_point_remap<VolumeWeighter>(old_mesh, new_mesh, 1, prod_dim,
          keys2edges, keys2prods, prods2new_ents, same_ents2old_ents,
          same_ents2new_ents, name);
    }
    for (auto& name : fields_to_remap[RemapType::PER_UNIT_MASS]) {
      swap_point_remap<MassWeighter>(old_mesh, new_mesh, 1, prod_dim,
          keys2edges, keys2prods, prods2new_ents, same_ents2old_ents,
          same_ents2new_ents, name);
    }
    remap_old_class_id(old_mesh, new_mesh, prods2new_ents, same_ents2old_ents,
        same_ents2new_ents);
  }
}

template <class Elem>
void Remap<Elem>::after_adapt() {
  Omega_h::ScopedTimer timer("Remap::after_adapt");
  sim.fields[sim.position].storage =
      Omega_h::deep_copy(sim.disc.get_node_coords());
  sim.fields.copy_from_omega_h(sim.disc, field_indices_to_remap);
  sim.fields.remove_from_omega_h(sim.disc, field_indices_to_remap);
  for (auto& name : fields_to_remap[RemapType::POSITIVE_DETERMINANT]) {
    auto const fi = sim.fields.find(name);
    auto const points_to_F = sim.getset(fi);
    auto const npoints = sim.fields[fi].support->count();
    auto functor = OMEGA_H_LAMBDA(int point) {
      auto log_F = getfull<Elem>(points_to_F, point);
      auto const F = Omega_h::exp_polar(log_F);
      OMEGA_H_CHECK(determinant(F) > 0.0);
      setfull<Elem>(points_to_F, point, F);
    };
    parallel_for(npoints, std::move(functor));
  }
}

template <class Elem>
RemapBase* remap_factory(Simulation& sim) {
  return new Remap<Elem>(sim);
}

#define LGR_EXPL_INST(Elem)                                                    \
  template RemapBase* remap_factory<Elem>(Simulation&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr
