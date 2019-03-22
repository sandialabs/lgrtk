#pragma once

#include <memory>
#include <vector>

#include <lgr_macros.hpp>
#include <lgr_vector3.hpp>
#include <lgr_host_vector.hpp>
#include <lgr_int_range.hpp>
#include <lgr_for_each.hpp>

namespace lgr {

struct all_space {
};

constexpr inline double distance(all_space const, vector3<double> const) {
  return 1.0;
};

struct plane {
  vector3<double> normal;
  double origin;
};

constexpr inline double distance(plane const pl, vector3<double> const pt) {
  return pl.normal * pt - pl.origin;
};

struct sphere {
  vector3<double> origin;
  double radius;
};

inline double distance(sphere const s, vector3<double> const pt) {
  return norm(pt - s.origin) - s.radius;
};

struct cylinder {
  vector3<double> axis;
  vector3<double> origin;
  double radius;
};

inline double distance(cylinder const s, vector3<double> const pt) {
  auto const pt_on_plane = pt - (pt * s.axis) * s.axis;
  auto const origin_on_plane = s.origin - (s.origin * s.axis) * s.axis;
  return s.radius - norm(pt_on_plane - origin_on_plane);
};

struct extruded_sine_wave {
  vector3<double> z_axis;
  vector3<double> x_axis;
  double z_offset;
  double sine_period;
  double sine_offset;
  double sine_amplitude;
};

inline double distance(extruded_sine_wave const w, vector3<double> const pt) {
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
    virtual void mark(host_vector<vector3<double>> const& points, int const marker, host_vector<int>* markers) const = 0;
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
  void mark(host_vector<vector3<double>> const& points, int const marker, host_vector<int>* markers) const override {
    int_range const range(points.size());
    auto const points_begin = points.cbegin();
    auto const markers_begin = markers->begin();
    auto const clips_begin = m_host_clips.cbegin();
    auto const clips_end = m_host_clips.cend();
    auto const source = m_source;
    auto functor = [=] (int const i) {
      vector3<double> const pt = points_begin[i];
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
};

class union_domain : public domain {
  std::vector<std::unique_ptr<domain>> m_domains;
  public:
  void add(std::unique_ptr<domain>&& uptr);
  void mark(host_vector<vector3<double>> const& points, int const marker, host_vector<int>* markers) const override;
};

}
