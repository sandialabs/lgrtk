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
  // the sum of these values in one element should be the reference element
  // volume they are used to integrate quantities over the reference element
  Vector<Elem::points> weights;
};

#ifdef LGR_BAR2
struct Bar2Side {
  static constexpr int dim = 1;
  static constexpr int nodes = 1;
  static constexpr int points = 1;
  static constexpr bool is_simplex = true;
  static OMEGA_H_INLINE Matrix<nodes, points> basis_values();
  static OMEGA_H_INLINE Vector<points> weights(Matrix<dim, nodes>);
  static OMEGA_H_INLINE constexpr double lumping(int const node);
};

struct Bar2 {
  static constexpr int dim = 1;
  static constexpr int nodes = 2;
  static constexpr int points = 1;
  static constexpr bool is_simplex = true;
  // given the reference positions of the nodes of one element,
  // return the ReferenceShape information
  static OMEGA_H_INLINE Shape<Bar2> shape(Matrix<dim, nodes> node_coords);
  static OMEGA_H_INLINE constexpr double lumping_factor(int /*node*/);
  static OMEGA_H_INLINE Matrix<nodes, points> basis_values();
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
  static OMEGA_H_INLINE Matrix<nodes, points> basis_values();
  static OMEGA_H_INLINE Vector<points> weights(Matrix<dim, nodes>);
  static OMEGA_H_INLINE constexpr double lumping(int const node);
};

struct Tri3 {
  static constexpr int dim = 2;
  static constexpr int nodes = 3;
  static constexpr int points = 1;
  static constexpr bool is_simplex = true;
  static OMEGA_H_INLINE Shape<Tri3> shape(Matrix<dim, nodes> node_coords);
  static OMEGA_H_INLINE constexpr double lumping_factor(int /*node*/);
  static OMEGA_H_INLINE Matrix<nodes, points> basis_values();
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
  static OMEGA_H_INLINE Matrix<nodes, points> basis_values();
  static OMEGA_H_INLINE Vector<points> weights(Matrix<dim, nodes>);
  static OMEGA_H_INLINE constexpr double lumping(int const node);
};

struct Tri6 {
 public:
  static constexpr int dim = 2;
  static constexpr int nodes = 6;
  static constexpr int points = 3;
  static constexpr bool is_simplex = true;

 private:
  static OMEGA_H_INLINE Matrix<dim, points> pts();
  static OMEGA_H_INLINE Matrix<dim, nodes> bgrads(Vector<dim> xi);
  static OMEGA_H_INLINE void compute_lengths(
      Matrix<dim, nodes> node_coords, Shape<Tri6>& shape);
  static OMEGA_H_INLINE void compute_gradients(
      Matrix<dim, nodes> node_coords, Shape<Tri6>& shape);

 public:
  static OMEGA_H_INLINE Shape<Tri6> shape(Matrix<dim, nodes> node_coords);
  static OMEGA_H_INLINE constexpr double lumping_factor(int const node);
  static OMEGA_H_INLINE Matrix<nodes, points> basis_values();
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
  static OMEGA_H_INLINE Matrix<nodes, points> basis_values();
  static OMEGA_H_INLINE Vector<points> weights(Matrix<dim, nodes>);
  static OMEGA_H_INLINE constexpr double lumping(int const node);
};

struct Quad4 {
 public:
  static constexpr int dim = 2;
  static constexpr int nodes = 4;
  static constexpr int points = 4;
  static constexpr bool is_simplex = false;

 private:
  static OMEGA_H_INLINE Matrix<2, 4> pts();
  static OMEGA_H_INLINE Matrix<2, 4> bgrads(Vector<2> xi);
  static OMEGA_H_INLINE void compute_lengths(
      Matrix<2, 4> node_coords, Shape<Quad4>& shape);
  static OMEGA_H_INLINE void compute_gradients(
      Matrix<2, 4> node_coords, Shape<Quad4>& shape);

 public:
  static OMEGA_H_INLINE Shape<Quad4> shape(Matrix<dim, nodes> node_coords);
  static OMEGA_H_INLINE constexpr double lumping_factor(int /* node */);
  static OMEGA_H_INLINE Matrix<nodes, points> basis_values();
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
  static OMEGA_H_INLINE Matrix<nodes, points> basis_values();
  static OMEGA_H_INLINE Vector<points> weights(Matrix<dim, nodes>);
  static OMEGA_H_INLINE constexpr double lumping(int const node);
};

struct Tet4 {
  static constexpr int dim = 3;
  static constexpr int nodes = 4;
  static constexpr int points = 1;
  static constexpr bool is_simplex = true;
  static OMEGA_H_INLINE Shape<Tet4> shape(Matrix<dim, nodes> node_coords);
  static OMEGA_H_INLINE constexpr double lumping_factor(int /*node*/);
  static OMEGA_H_INLINE Matrix<nodes, points> basis_values();
  static constexpr char const* name() { return "Tet4"; }
  using side = Tet4Side;
};
#endif

#ifdef LGR_COMPTET
// needed for side hack below
#ifndef LGR_TET4
#error Composite Tet requires LGR_TET4 ON
#endif
struct CompTet {
  static constexpr int dim = 3;
  static constexpr int nodes = 10;
  static constexpr int points = 4;
  static constexpr bool is_simplex = true;
  static constexpr int nsub_tets = 12;
  static constexpr int nbarycentric_coords = 4;

  static OMEGA_H_INLINE Shape<CompTet> shape(Matrix<dim, nodes> node_coords);

  static OMEGA_H_INLINE constexpr double lumping_factor(int /* node */);

  static OMEGA_H_INLINE Matrix<nodes, points> basis_values();
  static constexpr char const* name() { return "CompTet"; }

  using side = Tet4Side;  // this is wrong!
  using SType = Omega_h::Few<Omega_h::Matrix<nodes, dim>, nsub_tets>;
  using OType = Omega_h::Few<Omega_h::Matrix<dim, dim>, nsub_tets>;
  using STIPMType = Omega_h::Few<Omega_h::Matrix<4, 4>, nsub_tets>;
  using SOLType = Omega_h::Few<Omega_h::Matrix<nodes, dim>, 4>;

  template <int NI, int NJ, int NK>
  static OMEGA_H_INLINE void zero(
      Omega_h::Few<Omega_h::Matrix<NJ, NK>, NI>& in);

  static OMEGA_H_INLINE double det_44(Matrix<4, 4> a);

  static OMEGA_H_INLINE Matrix<4, 4> invert_44(Matrix<4, 4> a);

  static OMEGA_H_INLINE SType compute_S();

  static OMEGA_H_INLINE STIPMType compute_sub_tet_int_proj_M();

  static OMEGA_H_INLINE Matrix<nsub_tets, 4> compute_sub_tet_int();

  static OMEGA_H_INLINE Matrix<4, 4> compute_parent_M_inv();

  static OMEGA_H_INLINE OType compute_O(Matrix<dim, nodes> x, SType S);

  static OMEGA_H_INLINE OType compute_O_inv(OType O);

  static OMEGA_H_INLINE Vector<nsub_tets> compute_O_det(OType O);

  static OMEGA_H_INLINE Matrix<4, 4> compute_M_inv(Vector<nsub_tets> O_det);

  static OMEGA_H_INLINE Matrix<4, 4> compute_M_inv(Matrix<dim, nodes> node_coords);

  static OMEGA_H_INLINE SOLType compute_SOL(Vector<nsub_tets> O_det,
      OType O_inv, Matrix<nsub_tets, 4> sub_tet_int, SType S);

  static OMEGA_H_INLINE Vector<4> compute_DOL(
      Vector<nsub_tets> O_det, Matrix<nsub_tets, 4> sub_tet_int);

  static OMEGA_H_INLINE Matrix<dim, points> get_ref_points();

  static OMEGA_H_INLINE Vector<4> get_barycentric_coord(Vector<dim> x);

  static OMEGA_H_INLINE Matrix<dim, 4> get_subtet_coords(
      Matrix<dim, nodes> in, int subtet);

  static OMEGA_H_INLINE double compute_char_length(Matrix<dim, nodes> in);

};
#endif

#ifdef LGR_BAR2
#define LGR_EXPL_INST_BAR2 LGR_EXPL_INST(::lgr::Bar2)
#define LGR_EXPL_INST_BAR2_SIDE LGR_EXPL_INST(::lgr::Bar2::side)
#else
#define LGR_EXPL_INST_BAR2
#define LGR_EXPL_INST_BAR2_SIDE
#endif

#ifdef LGR_TRI3
#define LGR_EXPL_INST_TRI3 LGR_EXPL_INST(::lgr::Tri3)
#define LGR_EXPL_INST_TRI3_SIDE LGR_EXPL_INST(::lgr::Tri3::side)
#else
#define LGR_EXPL_INST_TRI3
#define LGR_EXPL_INST_TRI3_SIDE
#endif

#ifdef LGR_TRI6
#define LGR_EXPL_INST_TRI6 LGR_EXPL_INST(::lgr::Tri6)
#define LGR_EXPL_INST_TRI6_SIDE LGR_EXPL_INST(::lgr::Tri6::side)
#else
#define LGR_EXPL_INST_TRI6
#define LGR_EXPL_INST_TRI6_SIDE
#endif

#ifdef LGR_QUAD4
#define LGR_EXPL_INST_QUAD4 LGR_EXPL_INST(::lgr::Quad4)
#define LGR_EXPL_INST_QUAD4_SIDE LGR_EXPL_INST(::lgr::Quad4::side)
#else
#define LGR_EXPL_INST_QUAD4
#define LGR_EXPL_INST_QUAD4_SIDE
#endif

#ifdef LGR_TET4
#define LGR_EXPL_INST_TET4 LGR_EXPL_INST(::lgr::Tet4)
#define LGR_EXPL_INST_TET4_SIDE LGR_EXPL_INST(::lgr::Tet4::side)
#else
#define LGR_EXPL_INST_TET4
#define LGR_EXPL_INST_TET4_SIDE
#endif

#ifdef LGR_COMPTET
#define LGR_EXPL_INST_COMPTET LGR_EXPL_INST(::lgr::CompTet)
#else
#define LGR_EXPL_INST_COMPTET
#define LGR_EXPL_INST_COMPTET_SIDE
#endif

#define LGR_EXPL_INST_ELEMS                                                    \
  LGR_EXPL_INST_BAR2                                                           \
  LGR_EXPL_INST_TRI3                                                           \
  LGR_EXPL_INST_TRI6                                                           \
  LGR_EXPL_INST_QUAD4                                                          \
  LGR_EXPL_INST_TET4                                                           \
  LGR_EXPL_INST_COMPTET

#define LGR_EXPL_INST_ELEMS_AND_SIDES                                          \
  LGR_EXPL_INST_BAR2                                                           \
  LGR_EXPL_INST_BAR2_SIDE                                                      \
  LGR_EXPL_INST_TRI3                                                           \
  LGR_EXPL_INST_TRI3_SIDE                                                      \
  LGR_EXPL_INST_TRI6                                                           \
  LGR_EXPL_INST_TRI6_SIDE                                                      \
  LGR_EXPL_INST_QUAD4                                                          \
  LGR_EXPL_INST_QUAD4_SIDE                                                     \
  LGR_EXPL_INST_TET4                                                           \
  LGR_EXPL_INST_TET4_SIDE                                                      \
  LGR_EXPL_INST_COMPTET

}  // namespace lgr

#endif
