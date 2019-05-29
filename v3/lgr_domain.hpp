#pragma once

#include <memory>
#include <vector>

#include <lgr_macros.hpp>
#include <hpc_vector3.hpp>
#include <lgr_for_each.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_material_set.hpp>
#include <hpc_array_vector.hpp>

namespace lgr {

struct all_space {
};

constexpr inline double distance(all_space const, hpc::vector3<double> const) {
  return 1.0;
};

struct plane {
  hpc::vector3<double> normal;
  double origin;
};

constexpr inline double distance(plane const pl, hpc::vector3<double> const pt) {
  return pl.normal * pt - pl.origin;
};

struct sphere {
  hpc::vector3<double> origin;
  double radius;
};

inline double distance(sphere const s, hpc::vector3<double> const pt) {
  return s.radius - norm(pt - s.origin);
};

struct cylinder {
  hpc::vector3<double> axis;
  hpc::vector3<double> origin;
  double radius;
};

inline double distance(cylinder const s, hpc::vector3<double> const pt) {
  auto const pt_on_plane = pt - (pt * s.axis) * s.axis;
  auto const origin_on_plane = s.origin - (s.origin * s.axis) * s.axis;
  return s.radius - norm(pt_on_plane - origin_on_plane);
};

struct extruded_sine_wave {
  hpc::vector3<double> z_axis;
  hpc::vector3<double> x_axis;
  double z_offset;
  double sine_period;
  double sine_offset;
  double sine_amplitude;
};

inline double distance(extruded_sine_wave const w, hpc::vector3<double> const pt) {
  auto const proj = (pt * w.z_axis) * w.z_axis;
  auto const z = norm(proj) - w.z_offset;
  auto const rej = pt - proj;
  auto const x = rej * w.x_axis;
  auto const angle = (x - w.sine_offset) * ((2.0 * pi) / (w.sine_period));
  auto const z_zero = w.sine_amplitude * std::sin(angle);
  return z_zero - z;
}

class domain {
  public:
    domain() = default;
    domain(domain&&) = default;
    virtual ~domain();
    virtual void mark(
        hpc::device_array_vector<hpc::vector3<double>, node_index> const& points,
        int const marker,
        hpc::device_vector<int, node_index>* markers) const = 0;
    virtual void mark(
        hpc::device_array_vector<hpc::vector3<double>, element_index> const& points,
        material_index const marker,
        hpc::device_vector<material_index, element_index>* markers) const = 0;
    virtual void mark(
        hpc::device_array_vector<hpc::vector3<double>, node_index> const& points,
        material_index const marker,
        hpc::device_vector<material_set, node_index>* markers) const = 0;
};

template <class SourceDomain>
class clipped_domain : public domain {
  SourceDomain m_source;
  std::vector<plane> m_host_clips;
  public:
  explicit clipped_domain(SourceDomain const& s)
    :m_source(s)
  {}
  void clip(plane const& p) {
    m_host_clips.push_back(p);
  }
  template <class Index, class Marker>
  void mark_tmpl(
      hpc::device_array_vector<hpc::vector3<double>, Index> const& points,
      Marker const marker,
      hpc::device_vector<Marker, Index>* markers) const {
    hpc::counting_range<Index> const range(points.size());
    auto const points_begin = points.cbegin();
    auto const markers_begin = markers->begin();
    auto const clips_begin = m_host_clips.cbegin();
    auto const clips_end = m_host_clips.cend();
    auto const source = m_source;
    auto functor = [=] (Index const i) {
      auto const pt = points_begin[i].load();
      bool is_in = (distance(source, pt) >= 0.0);
      for (auto clips_it = clips_begin; is_in && (clips_it != clips_end); ++clips_it) {
        is_in &= (distance(*clips_it, pt) >= 0.0);
      }
      if (is_in) {
        markers_begin[i] = marker;
      }
    };
    lgr::for_each(range, functor);
  }
  void mark(
      hpc::device_array_vector<hpc::vector3<double>, node_index> const& points,
      int const marker,
      hpc::device_vector<int, node_index>* markers) const override {
    this->mark_tmpl<node_index, int>(points, marker, markers);
  }
  void mark(
      hpc::device_array_vector<hpc::vector3<double>, element_index> const& points,
      material_index const marker,
      hpc::device_vector<material_index, element_index>* markers) const override {
    this->mark_tmpl<element_index, material_index>(points, marker, markers);
  }
  void mark(
      hpc::device_array_vector<hpc::vector3<double>, node_index> const& points,
      material_index const marker,
      hpc::device_vector<material_set, node_index>* markers) const override {
    hpc::counting_range<node_index> const range(points.size());
    auto const points_begin = points.cbegin();
    auto const markers_begin = markers->begin();
    auto const clips_begin = m_host_clips.cbegin();
    auto const clips_end = m_host_clips.cend();
    auto const source = m_source;
    auto functor = [=] (node_index const i) {
      auto const pt = points_begin[i].load();
      bool is_in = (distance(source, pt) >= 0.0);
      for (auto clips_it = clips_begin; is_in && (clips_it != clips_end); ++clips_it) {
        is_in &= (distance(*clips_it, pt) >= 0.0);
      }
      if (is_in) {
        material_set set = markers_begin[i];
        set = set | material_set(marker);
        markers_begin[i] = set;
      }
    };
    lgr::for_each(range, functor);
  }
};

class union_domain : public domain {
  std::vector<std::unique_ptr<domain>> m_domains;
  public:
  void add(std::unique_ptr<domain>&& uptr);
  void mark(
      hpc::device_array_vector<hpc::vector3<double>, node_index> const& points,
      int const marker,
      hpc::device_vector<int, node_index>* markers) const override;
  void mark(
      hpc::device_array_vector<hpc::vector3<double>, element_index> const& points,
      material_index const marker,
      hpc::device_vector<material_index, element_index>* markers) const override;
  void mark(
      hpc::device_array_vector<hpc::vector3<double>, node_index> const& points,
      material_index const marker,
      hpc::device_vector<material_set, node_index>* markers) const override;
};

std::unique_ptr<domain> epsilon_around_plane_domain(plane const& p, double eps);
std::unique_ptr<domain> sphere_domain(hpc::vector3<double> const origin, double const radius);
std::unique_ptr<domain> half_space_domain(plane const& p);
std::unique_ptr<domain> box_domain(hpc::vector3<double> const lower_left, hpc::vector3<double> const upper_right);

class input;
class state;

void assign_element_materials(input const& in, state& s);
void compute_nodal_materials(input const& in, state& s);
void collect_element_sets(input const& in, state& s);
void collect_node_sets(input const& in, state& s);

}
