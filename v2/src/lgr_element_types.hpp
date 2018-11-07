#ifndef LGR_ELEMENT_TYPES_HPP
#define LGR_ELEMENT_TYPES_HPP

#include <lgr_config.hpp>
#include <lgr_math.hpp>

namespace lgr {

template <class Elem>
struct Lengths {
  double time_step_length;
  double viscosity_length;
};

template <class Elem>
struct Shape {
  Lengths<Elem> lengths;
  // at each integration point, the gradient with respect to reference space
  // of the nodal basis functions
  Omega_h::Few<Matrix<Elem::dim, Elem::nodes>, Elem::points> basis_gradients;
  // these values are |J| times the original integration point weights
  // the sum of these values in one element should be the reference element volume
  // they are used to integrate quantities over the reference element
  Vector<Elem::points> weights;
};

#ifdef LGR_BAR2
struct Bar2Side {
  static constexpr int dim = 1;
  static constexpr int nodes = 1;
  static constexpr int points = 1;
  static constexpr bool is_simplex = true;
  static OMEGA_H_INLINE Matrix<nodes, points> basis_values() {
    Matrix<nodes, points> out;
    out[0][0] = 1.0;
    return out;
  }
};

struct Bar2 {
  static constexpr int dim = 1;
  static constexpr int nodes = 2;
  static constexpr int points = 1;
  static constexpr bool is_simplex = true;
  // given the reference positions of the nodes of one element,
  // return the ReferenceShape information
  static OMEGA_H_INLINE
  Shape<Bar2> shape(
      Matrix<dim, nodes> node_coords) {
    Shape<Bar2> out;
    auto len = node_coords[1][0] - node_coords[0][0];
    out.weights[0] = len;
    auto inv_len = 1.0 / len;
    out.basis_gradients[0][0][0] = -inv_len;
    out.basis_gradients[0][1][0] = inv_len;
    out.lengths.time_step_length = len;
    out.lengths.viscosity_length = len;
    return out;
  }
  static OMEGA_H_INLINE
  constexpr double lumping_factor(int /*node*/) { return 1.0 / 2.0; }
  static OMEGA_H_INLINE Matrix<nodes, points> basis_values() {
    Matrix<nodes, points> out;
    out[0][0] = 1.0 / 2.0;
    out[0][1] = 1.0 / 2.0;
    return out;
  }
  static constexpr char const* name() { return "Bar2"; }
  using side = Bar2Side;
};
#endif

#ifdef LGR_TRI3
struct Tri3Side {
  static constexpr int dim = 2;
  static constexpr int nodes = 2;
  static constexpr int points = 1;
  static constexpr bool is_simplex = true;
  static OMEGA_H_INLINE Matrix<nodes, points> basis_values() {
    Matrix<nodes, points> out;
    out[0][0] = 1.0 / 2.0;
    out[0][1] = 1.0 / 2.0;
    return out;
  }
};

struct Tri3 {
  static constexpr int dim = 2;
  static constexpr int nodes = 3;
  static constexpr int points = 1;
  static constexpr bool is_simplex = true;
  static OMEGA_H_INLINE
  Shape<Tri3> shape(Matrix<dim, nodes> node_coords) {
    Matrix<2, 3> edge_vectors;
    edge_vectors[0] = node_coords[1] - node_coords[0];
    edge_vectors[1] = node_coords[2] - node_coords[0];
    edge_vectors[2] = node_coords[2] - node_coords[1];
    Vector<3> squared_edge_lengths;
    for (int i = 0; i < 3; ++i) squared_edge_lengths[i] = Omega_h::norm_squared(edge_vectors[i]);
    Shape<Tri3> out;
    auto max_squared_edge_length = Omega_h::reduce(squared_edge_lengths, Omega_h::maximum<double>());
    auto max_edge_length = std::sqrt(max_squared_edge_length);
    out.lengths.viscosity_length = max_edge_length;
    Matrix<2, 3> raw_gradients;
    raw_gradients[0] = Omega_h::perp(edge_vectors[2]);
    raw_gradients[1] = -Omega_h::perp(edge_vectors[1]);
    raw_gradients[2] = Omega_h::perp(edge_vectors[0]);
    auto raw_area = edge_vectors[0] * raw_gradients[1];
    out.weights[0] = raw_area * (1.0 / 2.0);
    auto inv_raw_area = 1.0 / raw_area;
    out.basis_gradients[0] = raw_gradients * inv_raw_area;
    auto min_height = raw_area / max_edge_length;
    out.lengths.time_step_length = min_height;
    return out;
  }
  static OMEGA_H_INLINE
  constexpr double lumping_factor(int /*node*/) { return 1.0 / 3.0; }
  static OMEGA_H_INLINE Matrix<nodes, points> basis_values() {
    Matrix<nodes, points> out;
    out[0][0] = 1.0 / 3.0;
    out[0][1] = 1.0 / 3.0;
    out[0][2] = 1.0 / 3.0;
    return out;
  }
  static constexpr char const* name() { return "Tri3"; }
  using side = Tri3Side;
};
#endif

#ifdef LGR_TRI6
struct Tri6Side {
  static constexpr int dim = 2;
  static constexpr int nodes = 3;
  static constexpr int points = 2;
  static constexpr bool is_simplex = true;
  static OMEGA_H_INLINE Matrix<nodes, points> basis_values() {
    Matrix<nodes, points> out;
    out[0][0] = 1.0 / 6.0 * (1.0 + std::sqrt(3.0));
    out[0][1] = 1.0 / 6.0 * (1.0 - std::sqrt(3.0));
    out[0][2] = 2.0 / 3.0;
    out[1][0] = 1.0 / 6.0 * (1.0 - std::sqrt(3.0));
    out[1][1] = 1.0 / 6.0 * (1.0 + std::sqrt(3.0));
    out[1][2] = 2.0 / 3.0;
    return out;
  }
};

struct Tri6 {
  public:
    static constexpr int dim = 2;
    static constexpr int nodes = 6;
    static constexpr int points = 3;
    static constexpr bool is_simplex = true;
  private:
    static OMEGA_H_INLINE Matrix<dim, points> pts() {
      Matrix<dim, points> out;
      out[0][0] = 2.0 / 3.0; out[0][1] = 1.0 / 6.0;
      out[1][0] = 1.0 / 6.0; out[1][1] = 2.0 / 3.0;
      out[2][0] = 1.0 / 6.0; out[2][1] = 1.0 / 6.0;
      return out;
    }
    static OMEGA_H_INLINE Matrix<dim, nodes> bgrads(Vector<dim> xi) {
      Matrix<dim, nodes> out;
      auto xi2 = 1.0 - xi[0] - xi[1];
      out[0][0] = -4.0 * xi2   + 1.0;   out[0][1] = -4.0 * xi2   + 1.0;
      out[1][0] =  4.0 * xi[0] - 1.0;   out[1][1] =  0.0;
      out[2][0] =  0.0;                 out[2][1] =  4.0 * xi[1] - 1.0;
      out[3][0] =  4.0 * (xi2 - xi[0]); out[3][1] = -4.0 * xi[0];
      out[4][0] =  4.0 * xi[1];         out[4][1] =  4.0 * xi[0];
      out[5][0] = -4.0 * xi[1];         out[5][1] =  4.0 * (xi2 - xi[1]);
      out[6][0] = 1.0;
      return out;
    }
    static OMEGA_H_INLINE void compute_lengths(
        Matrix<dim, nodes> node_coords, Shape<Tri6>& shape) {
      Matrix<2, 3> edge_vectors;
      edge_vectors[0] = node_coords[1] - node_coords[0];
      edge_vectors[1] = node_coords[2] - node_coords[0];
      edge_vectors[2] = node_coords[2] - node_coords[1];
      Vector<3> squared_edge_lengths;
      for (int i = 0; i < 3; ++i) {
        squared_edge_lengths[i] = Omega_h::norm_squared(edge_vectors[i]);
      }
      auto max_squared_edge_length =
        Omega_h::reduce(squared_edge_lengths, Omega_h::maximum<double>());
      auto max_edge_length = std::sqrt(max_squared_edge_length);
      shape.lengths.viscosity_length = max_edge_length;
      Matrix<2, 3> raw_gradients;
      raw_gradients[0] = Omega_h::perp(edge_vectors[2]);
      raw_gradients[1] = -Omega_h::perp(edge_vectors[1]);
      raw_gradients[2] = Omega_h::perp(edge_vectors[0]);
      auto raw_area = edge_vectors[0] * raw_gradients[1];
      auto min_height = raw_area / max_edge_length;
      shape.lengths.time_step_length = min_height;
    }
    static OMEGA_H_INLINE void compute_gradients(
        Matrix<dim, nodes> node_coords, Shape<Tri6>& shape) {
      auto ips = pts();
      auto x = Omega_h::transpose(node_coords);
      for (int ip = 0; ip < points; ++ip) {
        auto xi = ips[ip];
        auto dNdxi = bgrads(xi);
        auto J = dNdxi * x;
        auto Jinv = Omega_h::invert(J);
        auto dNdx = Jinv * dNdxi;
        shape.basis_gradients[ip] = dNdx;
        shape.weights[ip] = Omega_h::determinant(J) * 1.0 / 6.0;
      }
    }
  public:
    static OMEGA_H_INLINE
      Shape<Tri6> shape(Matrix<dim, nodes> node_coords) {
        Shape<Tri6> out;
        compute_lengths(node_coords, out);
        compute_gradients(node_coords, out);
        return out;
      }
    static OMEGA_H_INLINE
      constexpr double lumping_factor(int node) {
        // clang-format off
        return ((node == 0) ? 3.0 / 57.0 :
                (node == 1) ? 3.0 / 57.0 :
                (node == 2) ? 3.0 / 57.0 :
                (node == 3) ? 16.0 / 57.0 :
                (node == 4) ? 16.0 / 57.0 :
                (node == 5) ? 16.0 / 57.0 : -1.0);
        // clang-format on
      }
    static OMEGA_H_INLINE Matrix<nodes, points> basis_values() {
      Matrix<nodes, points> out;
      out[0][0] = -1.0 / 9.0;
      out[0][1] =  2.0 / 9.0;
      out[0][2] = -1.0 / 9.0;
      out[0][3] =  4.0 / 9.0;
      out[0][4] =  4.0 / 9.0;
      out[0][5] =  1.0 / 9.0;
      out[1][0] = -1.0 / 9.0;
      out[1][1] = -1.0 / 9.0;
      out[1][2] =  2.0 / 9.0;
      out[1][3] =  1.0 / 9.0;
      out[1][4] =  4.0 / 9.0;
      out[1][5] =  4.0 / 9.0;
      out[2][0] =  2.0 / 9.0;
      out[2][1] = -1.0 / 9.0;
      out[2][2] = -1.0 / 9.0;
      out[2][3] =  4.0 / 9.0;
      out[2][4] =  1.0 / 9.0;
      out[2][5] =  4.0 / 9.0;
      return out;
    }
    static constexpr char const* name() { return "Tri6"; }
    using side = Tri6Side;
};
#endif

#ifdef LGR_QUAD4
struct Quad4Side {
  static constexpr int dim = 2;
  static constexpr int nodes = 2;
  static constexpr int points = 1;
  static constexpr bool is_simplex = true;
  static OMEGA_H_INLINE Matrix<nodes, points> basis_values() {
    Matrix<nodes, points> out;
    out[0][0] = 1.0 / 2.0;
    out[0][1] = 1.0 / 2.0;
    return out;
  }
};

struct Quad4 {
  public:
    static constexpr int dim = 2;
    static constexpr int nodes = 4;
    static constexpr int points = 4;
    static constexpr bool is_simplex = false;
  private:
    static OMEGA_H_INLINE Matrix<2, 4> pts() {
      Matrix<2, 4> out;
      out[0][0] = -1.0 / std::sqrt(3.0); out[0][1] = -1.0 / std::sqrt(3.0);
      out[1][0] =  1.0 / std::sqrt(3.0); out[1][1] = -1.0 / std::sqrt(3.0);
      out[2][0] =  1.0 / std::sqrt(3.0); out[2][1] =  1.0 / std::sqrt(3.0);
      out[3][0] = -1.0 / std::sqrt(3.0); out[3][1] =  1.0 / std::sqrt(3.0);
      return out;
    }
    static OMEGA_H_INLINE Matrix<2, 4> bgrads(Vector<2> xi) {
      Matrix<2, 4> out;
      out[0][0] = -0.25 * (1.0 - xi[1]); out[0][1] = -0.25 * (1.0 - xi[0]);
      out[1][0] =  0.25 * (1.0 - xi[1]); out[1][1] = -0.25 * (1.0 - xi[0]);
      out[2][0] =  0.25 * (1.0 + xi[1]); out[2][1] =  0.25 * (1.0 + xi[0]);
      out[3][0] = -0.25 * (1.0 + xi[1]); out[3][1] =  0.25 * (1.0 + xi[0]);
      return out;
    }
    static OMEGA_H_INLINE void compute_lengths(
        Matrix<2, 4> node_coords, Shape<Quad4>& shape) {
      Matrix<2, 4> edge_vectors;
      edge_vectors[0] = node_coords[1] - node_coords[0];
      edge_vectors[1] = node_coords[2] - node_coords[1];
      edge_vectors[2] = node_coords[3] - node_coords[2];
      edge_vectors[3] = node_coords[0] - node_coords[3];
      Vector<4> squared_edge_lengths;
      for (int i = 0; i < 4; ++i) {
        squared_edge_lengths[i] = Omega_h::norm_squared(edge_vectors[i]);
      }
      auto max_squared_edge_length =
        Omega_h::reduce(squared_edge_lengths, Omega_h::maximum<double>());
      auto min_squared_edge_length =
        Omega_h::reduce(squared_edge_lengths, Omega_h::minimum<double>());
      auto max_edge_length = std::sqrt(max_squared_edge_length);
      auto min_edge_length = std::sqrt(min_squared_edge_length);
      shape.lengths.viscosity_length = max_edge_length;
      shape.lengths.time_step_length = min_edge_length;
    }
    static OMEGA_H_INLINE void compute_gradients(
        Matrix<2, 4> node_coords, Shape<Quad4>& shape) {
      Matrix<2, 4> ips = pts();
      Matrix<4, 2> x = Omega_h::transpose(node_coords);
      for (int ip = 0; ip < 4; ++ip) {
        Vector<2> xi = ips[ip];
        Matrix<2, 4> dNdxi = bgrads(xi);
        Matrix<2, 2> J = dNdxi * x;
        Matrix<2, 2> Jinv = Omega_h::invert(J);
        Matrix<2, 4> dNdx = Jinv * dNdxi;
        shape.basis_gradients[ip] = dNdx;
        shape.weights[ip] = Omega_h::determinant(J);
      }
    }
  public:
    static OMEGA_H_INLINE
    Shape<Quad4> shape(Matrix<dim, nodes> node_coords) {
      Shape<Quad4> out;
      compute_lengths(node_coords, out);
      compute_gradients(node_coords, out);
      return out;
    }
    static OMEGA_H_INLINE
    constexpr double lumping_factor(int /* node */) { return 1.0 / 4.0; }
    static OMEGA_H_INLINE Matrix<nodes, points> basis_values() {
      Matrix<nodes, points> out;
      out[0][0] = 1.0 / 6.0 * (2.0 + std::sqrt(3.0));
      out[0][1] = 1.0 / 6.0;
      out[0][2] = 1.0 / 6.0 * (2.0 - std::sqrt(3.0));
      out[0][3] = 1.0 / 6.0;
      out[1][0] = 1.0 / 6.0;
      out[1][1] = 1.0 / 6.0 * (2.0 + std::sqrt(3.0));
      out[1][2] = 1.0 / 6.0;
      out[1][3] = 1.0 / 6.0 * (2.0 - std::sqrt(3.0));
      out[2][0] = 1.0 / 6.0 * (2.0 - std::sqrt(3.0));
      out[2][1] = 1.0 / 6.0;
      out[2][2] = 1.0 / 6.0 * (2.0 + std::sqrt(3.0));
      out[2][3] = 1.0 / 6.0;
      out[3][0] = 1.0 / 6.0;
      out[3][1] = 1.0 / 6.0 * (2.0 - std::sqrt(3.0));
      out[3][2] = 1.0 / 6.0;
      out[3][3] = 1.0 / 6.0 * (2.0 + std::sqrt(3.0));
      return out;
    }
    static constexpr char const* name() { return "Quad4"; }
    using side = Quad4Side;
};
#endif

#ifdef LGR_TET4
struct Tet4Side {
  static constexpr int dim = 3;
  static constexpr int nodes = 3;
  static constexpr int points = 1;
  static constexpr bool is_simplex = true;
  static OMEGA_H_INLINE Matrix<nodes, points> basis_values() {
    Matrix<nodes, points> out;
    out[0][0] = 1.0 / 3.0;
    out[0][1] = 1.0 / 3.0;
    out[0][1] = 1.0 / 3.0;
    return out;
  }
};

struct Tet4 {
  static constexpr int dim = 3;
  static constexpr int nodes = 4;
  static constexpr int points = 1;
  static constexpr bool is_simplex = true;
  static OMEGA_H_INLINE
  Shape<Tet4> shape(Matrix<dim, nodes> node_coords) {
    Matrix<3, 6> edge_vectors;
    edge_vectors[0] = node_coords[1] - node_coords[0];
    edge_vectors[1] = node_coords[2] - node_coords[0];
    edge_vectors[2] = node_coords[3] - node_coords[0];
    edge_vectors[3] = node_coords[2] - node_coords[1];
    edge_vectors[4] = node_coords[3] - node_coords[1];
    edge_vectors[5] = node_coords[3] - node_coords[2];
    Vector<6> squared_edge_lengths;
    for (int i = 0; i < 6; ++i) {
      squared_edge_lengths[i] = Omega_h::norm_squared(edge_vectors[i]);
    }
    Shape<Tet4> out;
    auto max_squared_edge_length = Omega_h::reduce(squared_edge_lengths, Omega_h::maximum<double>());
    out.lengths.viscosity_length = std::sqrt(max_squared_edge_length);
    Matrix<3, 4> raw_gradients;
    // first compute "raw" gradients (gradients times volume times 6)
    raw_gradients[0] = Omega_h::cross(edge_vectors[4], edge_vectors[3]);
    raw_gradients[1] = Omega_h::cross(edge_vectors[1], edge_vectors[2]);
    raw_gradients[2] = Omega_h::cross(edge_vectors[2], edge_vectors[0]);
    raw_gradients[3] = Omega_h::cross(edge_vectors[0], edge_vectors[1]);
    auto raw_volume = raw_gradients[3] * edge_vectors[2];
    out.weights[0] = raw_volume * (1.0 / 6.0);
    auto inv_raw_volume = 1.0 / raw_volume;
    auto raw_volume_squared = square(raw_volume);
    Vector<4> squared_heights;
    for (int i = 0; i < 4; ++i) {
      // then convert "raw" gradients to true gradients by "dividing" by raw volume
      out.basis_gradients[0][i] = raw_gradients[i] * inv_raw_volume;
      auto raw_opposite_area_squared = Omega_h::norm_squared(raw_gradients[i]);
      squared_heights[i] = raw_volume_squared / raw_opposite_area_squared;
    }
    auto min_height_squared = Omega_h::reduce(squared_heights, Omega_h::minimum<double>());
    out.lengths.time_step_length = std::sqrt(min_height_squared);
    return out;
  }
  static OMEGA_H_INLINE
  constexpr double lumping_factor(int /*node*/) { return 1.0 / 4.0; }
  static OMEGA_H_INLINE Matrix<nodes, points> basis_values() {
    Matrix<nodes, points> out;
    out[0][0] = 1.0 / 4.0;
    out[0][1] = 1.0 / 4.0;
    out[0][2] = 1.0 / 4.0;
    out[0][3] = 1.0 / 4.0;
    return out;
  }
  static constexpr char const* name() { return "Tet4"; }
  using side = Tet4Side;
};
#endif

struct CompositeTet {
  static constexpr int dim = 3;
  static constexpr int nodes = 10;
  static constexpr int points = 4;
  static constexpr bool is_simplex = true;
};

#ifdef LGR_BAR2
#define LGR_EXPL_INST_BAR2 LGR_EXPL_INST(Bar2)
#define LGR_EXPL_INST_BAR2_SIDE LGR_EXPL_INST(Bar2::side)
#else
#define LGR_EXPL_INST_BAR2
#define LGR_EXPL_INST_BAR2_SIDE
#endif

#ifdef LGR_TRI3
#define LGR_EXPL_INST_TRI3 LGR_EXPL_INST(Tri3)
#define LGR_EXPL_INST_TRI3_SIDE LGR_EXPL_INST(Tri3::side)
#else
#define LGR_EXPL_INST_TRI3
#define LGR_EXPL_INST_TRI3_SIDE
#endif

#ifdef LGR_TRI6
#define LGR_EXPL_INST_TRI6 LGR_EXPL_INST(Tri6)
#define LGR_EXPL_INST_TRI6_SIDE LGR_EXPL_INST(Tri6::side)
#else
#define LGR_EXPL_INST_TRI6
#define LGR_EXPL_INST_TRI6_SIDE
#endif

#ifdef LGR_QUAD4
#define LGR_EXPL_INST_QUAD4 LGR_EXPL_INST(Quad4)
#define LGR_EXPL_INST_QUAD4_SIDE LGR_EXPL_INST(Quad4::side)
#else
#define LGR_EXPL_INST_QUAD4
#define LGR_EXPL_INST_QUAD4_SIDE
#endif

#ifdef LGR_TET4
#define LGR_EXPL_INST_TET4 LGR_EXPL_INST(Tet4)
#define LGR_EXPL_INST_TET4_SIDE LGR_EXPL_INST(Tet4::side)
#else
#define LGR_EXPL_INST_TET4
#define LGR_EXPL_INST_TET4_SIDE
#endif

#define LGR_EXPL_INST_ELEMS \
LGR_EXPL_INST_BAR2 \
LGR_EXPL_INST_TRI3 \
LGR_EXPL_INST_TRI6 \
LGR_EXPL_INST_QUAD4 \
LGR_EXPL_INST_TET4

#define LGR_EXPL_INST_ELEMS_AND_SIDES \
LGR_EXPL_INST_BAR2 \
LGR_EXPL_INST_BAR2_SIDE \
LGR_EXPL_INST_TRI3 \
LGR_EXPL_INST_TRI3_SIDE \
LGR_EXPL_INST_TRI6 \
LGR_EXPL_INST_TRI6_SIDE \
LGR_EXPL_INST_QUAD4 \
LGR_EXPL_INST_QUAD4_SIDE \
LGR_EXPL_INST_TET4 \
LGR_EXPL_INST_TET4_SIDE

}

#endif
