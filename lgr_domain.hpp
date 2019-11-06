#pragma once

#include <memory>
#include <vector>

#include <hpc_vector3.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_material_set.hpp>
#include <hpc_array_vector.hpp>
#include <hpc_numeric.hpp>
#include <hpc_dimensional.hpp>

namespace lgr {

struct all_space {
};

HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr hpc::length<double> distance(all_space const, hpc::position<double> const) noexcept {
  return 1.0;
};

struct plane {
  hpc::vector3<double> normal;
  hpc::length<double> origin;
};

HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr hpc::length<double> distance(plane const pl, hpc::position<double> const pt) noexcept {
  return pl.normal * pt - pl.origin;
};

struct sphere {
  hpc::position<double> origin;
  hpc::length<double> radius;
};

HPC_ALWAYS_INLINE HPC_HOST_DEVICE hpc::length<double> distance(sphere const s, hpc::position<double> const pt) noexcept {
  return s.radius - norm(pt - s.origin);
};

struct cylinder {
  hpc::vector3<double> axis;
  hpc::position<double> origin;
  hpc::length<double> radius;
};

HPC_ALWAYS_INLINE HPC_HOST_DEVICE hpc::length<double> distance(cylinder const s, hpc::position<double> const pt) noexcept {
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

HPC_ALWAYS_INLINE HPC_HOST_DEVICE double distance(extruded_sine_wave const w, hpc::vector3<double> const pt) noexcept {
  auto const proj = (pt * w.z_axis) * w.z_axis;
  auto const z = norm(proj) - w.z_offset;
  auto const rej = pt - proj;
  auto const x = rej * w.x_axis;
  auto const angle = (x - w.sine_offset) * ((2.0 * hpc::pi<double>()) / (w.sine_period));
  auto const z_zero = w.sine_amplitude * std::sin(angle);
  return z_zero - z;
}

class domain {
  public:
    domain() = default;
    domain(domain&&) = default;
    virtual ~domain();
    virtual void mark(
        hpc::device_array_vector<hpc::position<double>, node_index> const& points,
        int const marker,
        hpc::device_vector<int, node_index>* markers) const = 0;
#ifdef HPC_ENABLE_STRONG_INDICES
    virtual void mark(
        hpc::device_array_vector<hpc::position<double>, element_index> const& points,
        material_index const marker,
        hpc::device_vector<material_index, element_index>* markers) const = 0;
#endif
    virtual void mark(
        hpc::device_array_vector<hpc::position<double>, node_index> const& points,
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
      hpc::device_array_vector<hpc::position<double>, Index> const& points,
      Marker const marker,
      hpc::device_vector<Marker, Index>* markers) const {
    hpc::counting_range<Index> const range(points.size());
    hpc::pinned_vector<plane, std::size_t> pinned_clips(m_host_clips.size());
    hpc::copy(hpc::serial_policy(), m_host_clips, pinned_clips);
    hpc::device_vector<plane, std::size_t> device_clips(m_host_clips.size());
    hpc::copy(pinned_clips, device_clips);
    auto const clips_range = hpc::make_iterator_range(device_clips.begin(), device_clips.end());
    auto const points_begin = points.cbegin();
    auto const markers_begin = markers->begin();
    auto const source = m_source;
    auto functor = [=] HPC_DEVICE (Index const i) {
      auto const pt = points_begin[i].load();
      bool is_in = (distance(source, pt) >= 0.0);
      for (auto const clip_plane : clips_range) {
        is_in &= (distance(clip_plane, pt) >= 0.0);
      }
      if (is_in) {
        markers_begin[i] = marker;
      }
    };
    hpc::for_each(hpc::device_policy(), range, functor);
  }
  void mark(
      hpc::device_array_vector<hpc::position<double>, node_index> const& points,
      int const marker,
      hpc::device_vector<int, node_index>* markers) const override {
    this->mark_tmpl<node_index, int>(points, marker, markers);
  }
#ifdef HPC_ENABLE_STRONG_INDICES
  void mark(
      hpc::device_array_vector<hpc::position<double>, element_index> const& points,
      material_index const marker,
      hpc::device_vector<material_index, element_index>* markers) const override {
    this->mark_tmpl<element_index, material_index>(points, marker, markers);
  }
#endif
  void mark(
      hpc::device_array_vector<hpc::position<double>, node_index> const& points,
      material_index const marker,
      hpc::device_vector<material_set, node_index>* markers) const override {
    hpc::counting_range<node_index> const range(points.size());
    hpc::pinned_vector<plane, std::size_t> pinned_clips(m_host_clips.size());
    hpc::copy(hpc::serial_policy(), m_host_clips, pinned_clips);
    hpc::device_vector<plane, std::size_t> device_clips(m_host_clips.size());
    hpc::copy(pinned_clips, device_clips);
    auto const clips_range = hpc::make_iterator_range(device_clips.begin(), device_clips.end());
    auto const points_begin = points.cbegin();
    auto const markers_begin = markers->begin();
    auto const source = m_source;
    auto functor = [=] HPC_DEVICE (node_index const i) {
      auto const pt = points_begin[i].load();
      bool is_in = (distance(source, pt) >= 0.0);
      for (auto const clip_plane : clips_range) {
        is_in &= (distance(clip_plane, pt) >= 0.0);
      }
      if (is_in) {
        material_set set = markers_begin[i];
        set = set | material_set(marker);
        markers_begin[i] = set;
      }
    };
    hpc::for_each(hpc::device_policy(), range, functor);
  }
};

class union_domain : public domain {
  std::vector<std::unique_ptr<domain>> m_domains;
  public:
  void add(std::unique_ptr<domain>&& uptr);
  void mark(
      hpc::device_array_vector<hpc::position<double>, node_index> const& points,
      int const marker,
      hpc::device_vector<int, node_index>* markers) const override;
#ifdef HPC_ENABLE_STRONG_INDICES
  void mark(
      hpc::device_array_vector<hpc::position<double>, element_index> const& points,
      material_index const marker,
      hpc::device_vector<material_index, element_index>* markers) const override;
#endif
  void mark(
      hpc::device_array_vector<hpc::position<double>, node_index> const& points,
      material_index const marker,
      hpc::device_vector<material_set, node_index>* markers) const override;
};

std::unique_ptr<domain> epsilon_around_plane_domain(plane const& p, double eps);
std::unique_ptr<domain> sphere_domain(hpc::position<double> const origin, double const radius);
std::unique_ptr<domain> half_space_domain(plane const& p);
std::unique_ptr<domain> box_domain(hpc::position<double> const lower_left, hpc::position<double> const upper_right);

class input;
class state;

void assign_element_materials(input const& in, state& s);
void compute_nodal_materials(input const& in, state& s);
void collect_element_sets(input const& in, state& s);
void collect_node_sets(input const& in, state& s);

}
