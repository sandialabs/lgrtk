#include <lgr_stabilized.hpp>
#include <lgr_state.hpp>
#include <lgr_for_each.hpp>
#include <lgr_input.hpp>

namespace lgr {

void update_p_h(state& s, double const dt,
    material_index const material,
    device_vector<double, node_index> const& old_p_h_vector) {
  auto const nodes_to_p_h = s.p_h[material].begin();
  auto const nodes_to_old_p_h = old_p_h_vector.cbegin();
  auto const nodes_to_p_h_dot = s.p_h_dot[material].cbegin();
  auto functor = [=] (node_index const node) {
    double const old_p_h = nodes_to_old_p_h[node];
    double const p_h_dot = nodes_to_p_h_dot[node];
    double const p_h = old_p_h + dt * p_h_dot;
    nodes_to_p_h[node] = p_h;
  };
  lgr::for_each(s.node_sets[material], functor);
}

void update_e_h(state& s, double const dt,
    material_index const material,
    device_vector<double, node_index> const& old_e_h_vector)
{
  auto const nodes_to_e_h_dot = s.e_h_dot[material].cbegin();
  auto const nodes_to_old_e_h = old_e_h_vector.cbegin();
  auto const nodes_to_e_h = s.e_h[material].begin();
  auto functor = [=] (node_index const node) {
    auto const e_h_dot = nodes_to_e_h_dot[node];
    double const old_e_h = nodes_to_old_e_h[node];
    auto const e_h = old_e_h + dt * e_h_dot;
    nodes_to_e_h[node] = e_h;
  };
  lgr::for_each(s.node_sets[material], functor);
}

void update_sigma_with_p_h(state& s, material_index const material) {
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_element_points = s.elements * s.points_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_in_element = s.nodes_in_element;
  auto const nodes_to_p_h = s.p_h[material].cbegin();
  auto const N = 1.0 / double(int(s.nodes_in_element.size()));
  auto const points_to_sigma = s.sigma.begin();
  auto functor = [=] (element_index const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    auto const element_points = elements_to_element_points[element];
    for (auto const point : element_points) {
      double point_p_h = 0.0;
      for (auto const node_in_element : nodes_in_element) {
        auto const element_node = element_nodes[node_in_element];
        auto const node = element_nodes_to_nodes[element_node];
        double const p_h = nodes_to_p_h[node];
        point_p_h = point_p_h + N * p_h;
      }
      symmetric3x3<double> const old_sigma = points_to_sigma[point];
      auto const new_sigma = deviator(old_sigma) - point_p_h;
      points_to_sigma[point] = new_sigma;
    }
  };
  lgr::for_each(s.element_sets[material], functor);
}

static void LGR_NOINLINE update_v_prime(input const& in, state& s, material_index const material)
{
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const points_to_dt = s.element_dt.cbegin();
  auto const points_to_rho = s.rho.cbegin();
  auto const nodes_to_a = s.a.cbegin();
  auto const nodes_to_p_h = s.p_h[material].cbegin();
  auto const points_to_v_prime = s.v_prime.begin();
  auto const c_tau = in.c_tau[material];
  auto const N = 1.0 / double(int(s.nodes_in_element.size()));
  auto functor = [=] (element_index const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    for (auto const point : elements_to_points[element]) {
      auto const point_nodes = points_to_point_nodes[point];
      double const dt = points_to_dt[point];
      auto const tau = c_tau * dt;
      auto grad_p = vector3<double>::zero();
      auto a = vector3<double>::zero();
      for (auto const node_in_element : nodes_in_element) {
        auto const element_node = element_nodes[node_in_element];
        auto const point_node = point_nodes[node_in_element];
        node_index const node = element_nodes_to_nodes[element_node];
        double const p_h = nodes_to_p_h[node];
        vector3<double> const grad_N = point_nodes_to_grad_N[point_node];
        grad_p = grad_p + (grad_N * p_h);
        vector3<double> const a_of_node = nodes_to_a[node];
        a = a + a_of_node;
      }
      a = a * N;
      double const rho = points_to_rho[point];
      auto const v_prime = -(tau / rho) * (rho * a + grad_p);
      points_to_v_prime[point] = v_prime;
    }
  };
  lgr::for_each(s.element_sets[material], functor);
}

static void LGR_NOINLINE update_q(input const& in, state& s, material_index const material)
{
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const points_to_dt = s.element_dt.cbegin();
  auto const points_to_rho = s.rho.cbegin();
  auto const nodes_to_a = s.a.cbegin();
  auto const nodes_to_p_h = s.p_h[material].cbegin();
  auto const points_to_q = s.q.begin();
  auto const c_tau = in.c_tau[material];
  auto const N = 1.0 / double(int(s.nodes_in_element.size()));
  auto functor = [=] (element_index const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    for (auto const point : elements_to_points[element]) {
      double const dt = points_to_dt[point];
      auto const tau = c_tau * dt;
      auto grad_p = vector3<double>::zero();
      auto a = vector3<double>::zero();
      double p_h = 0.0;
      auto const point_nodes = points_to_point_nodes[point];
      for (auto const node_in_element : nodes_in_element) {
        auto const element_node = element_nodes[node_in_element];
        auto const point_node = point_nodes[node_in_element];
        node_index const node = element_nodes_to_nodes[element_node];
        double const p_h_of_node = nodes_to_p_h[node];
        p_h = p_h + p_h_of_node;
        vector3<double> const grad_N = point_nodes_to_grad_N[point_node];
        grad_p = grad_p + (grad_N * p_h_of_node);
        vector3<double> const a_of_node = nodes_to_a[node];
        a = a + a_of_node;
      }
      a = a * N;
      p_h = p_h * N;
      double const rho = points_to_rho[point];
      auto const v_prime = -(tau / rho) * (rho * a + grad_p);
      auto const q = p_h * v_prime;
      points_to_q[point] = q;
    }
  };
  lgr::for_each(s.element_sets[material], functor);
}

static void LGR_NOINLINE update_p_h_W(state& s, material_index const material)
{
  auto const points_to_K = s.K.cbegin();
  auto const points_to_v_prime = s.v_prime.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const points_to_symm_grad_v = s.symm_grad_v.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const point_nodes_to_W = s.W.begin();
  double const N = 1.0 / double(int(s.nodes_in_element.size()));
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    for (auto const point : elements_to_points[element]) {
      symmetric3x3<double> symm_grad_v = points_to_symm_grad_v[point];
      double const div_v = trace(symm_grad_v);
      double const K = points_to_K[point];
      double const V = points_to_V[point];
      vector3<double> const v_prime = points_to_v_prime[point];
      auto const point_nodes = points_to_point_nodes[point];
      for (auto const point_node : point_nodes) {
        vector3<double> const grad_N = point_nodes_to_grad_N[point_node];
        double const p_h_dot =
          -(N * (K * div_v)) + (grad_N * (K * v_prime));
        double const W = p_h_dot * V;
        point_nodes_to_W[point_node] = W;
      }
    }
  };
  for_each(s.element_sets[material], functor);
}

static void LGR_NOINLINE update_e_h_W(state& s, material_index const material)
{
  auto const points_to_q = s.q.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const points_to_rho_e_dot = s.rho_e_dot.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const point_nodes_to_W = s.W.begin();
  double const N = 1.0 / double(int(s.nodes_in_element.size()));
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    for (auto const point : elements_to_points[element]) {
      double const rho_e_dot = points_to_rho_e_dot[point];
      double const V = points_to_V[point];
      vector3<double> const q = points_to_q[point];
      auto const point_nodes = points_to_point_nodes[point];
      for (auto const point_node : point_nodes) {
        vector3<double> const grad_N = point_nodes_to_grad_N[point_node];
        double const rho_e_h_dot = (N * rho_e_dot) + (grad_N * q);
        double const W = rho_e_h_dot * V;
        point_nodes_to_W[point_node] = W;
      }
    }
  };
  for_each(s.element_sets[material], functor);
}

static void LGR_NOINLINE update_p_h_dot(state& s, material_index const material)
{
  auto const nodes_to_node_elements = s.nodes_to_node_elements.cbegin();
  auto const node_elements_to_elements = s.node_elements_to_elements.cbegin();
  auto const node_elements_to_nodes_in_element = s.node_elements_to_nodes_in_element.cbegin();
  auto const point_nodes_to_W = s.W.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const nodes_to_p_h_dot = s.p_h_dot[material].begin();
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto const elements_to_material = s.material.cbegin();
  double const N = 1.0 / double(int(s.nodes_in_element.size()));
  auto functor = [=] (node_index const node) {
    double node_W = 0.0;
    double node_V = 0.0;
    auto const node_elements = nodes_to_node_elements[node];
    for (auto const node_element : node_elements) {
      auto const element = node_elements_to_elements[node_element];
      material_index const element_material = elements_to_material[element];
      if (element_material != material) continue;
      auto const node_in_element = node_elements_to_nodes_in_element[node_element];
      for (auto const point : elements_to_points[element]) {
        auto const point_nodes = points_to_point_nodes[point];
        auto const point_node = point_nodes[node_in_element];
        double const W = point_nodes_to_W[point_node];
        double const V = points_to_V[point];
        node_W = node_W + W;
        node_V = node_V + (N * V);
      }
    }
    auto const p_h_dot = node_W / node_V;
    nodes_to_p_h_dot[node] = p_h_dot;
  };
  lgr::for_each(s.node_sets[material], functor);
}

static void LGR_NOINLINE update_e_h_dot(state& s, material_index const material)
{
  auto const nodes_to_node_elements = s.nodes_to_node_elements.cbegin();
  auto const node_elements_to_elements = s.node_elements_to_elements.cbegin();
  auto const node_elements_to_nodes_in_element = s.node_elements_to_nodes_in_element.cbegin();
  auto const point_nodes_to_W = s.W.cbegin();
  auto const nodes_to_e_h_dot = s.e_h_dot[material].begin();
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto const nodes_to_m = s.material_mass[material].cbegin();
  auto const elements_to_material = s.material.cbegin();
  auto functor = [=] (node_index const node) {
    double node_W = 0.0;
    auto const node_elements = nodes_to_node_elements[node];
    for (auto const node_element : node_elements) {
      element_index const element = node_elements_to_elements[node_element];
      material_index const element_material = elements_to_material[element];
      if (element_material != material) continue;
      node_in_element_index const node_in_element = node_elements_to_nodes_in_element[node_element];
      for (auto const point : elements_to_points[element]) {
        auto const point_nodes = points_to_point_nodes[point];
        auto const point_node = point_nodes[node_in_element];
        double const W = point_nodes_to_W[point_node];
        node_W = node_W + W;
      }
    }
    double const m = nodes_to_m[node];
    auto const e_h_dot = node_W / m;
    nodes_to_e_h_dot[node] = e_h_dot;
  };
  lgr::for_each(s.node_sets[material], functor);
}

void nodal_ideal_gas(input const& in, state& s, material_index const material) {
  auto const nodes_to_rho = s.rho_h[material].cbegin();
  auto const nodes_to_e = s.e_h[material].cbegin();
  fill(s.p_h[material], double(0.0));
  auto const nodes_to_p = s.p_h[material].begin();
  auto const nodes_to_K = s.K_h[material].begin();
  auto const gamma = in.gamma[material];
  auto functor = [=] (node_index const node) {
    double const rho = nodes_to_rho[node];
    assert(rho > 0.0);
    double const e = nodes_to_e[node];
    assert(e > 0.0);
    auto const p = (gamma - 1.0) * (rho * e);
    assert(p > 0.0);
    nodes_to_p[node] = p;
    auto const K = gamma * p;
    nodes_to_K[node] = K;
  };
  lgr::for_each(s.node_sets[material], functor);
}

void update_nodal_density(state& s, material_index const material)
{
  auto const nodes_to_node_elements = s.nodes_to_node_elements.cbegin();
  auto const node_elements_to_elements = s.node_elements_to_elements.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const nodes_to_m = s.material_mass[material].cbegin();
  fill(s.rho_h[material], double(0.0));
  auto const nodes_to_rho_h = s.rho_h[material].begin();
  auto const N = 1.0 / double(int(s.nodes_in_element.size()));
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const elements_to_material = s.material.cbegin();
  auto functor = [=] (node_index const node) {
    double node_V(0.0);
    auto const node_elements = nodes_to_node_elements[node];
    for (auto const node_element : node_elements) {
      element_index const element = node_elements_to_elements[node_element];
      material_index const element_material = elements_to_material[element];
      if (element_material != material) continue;
      for (auto const point : elements_to_points[element]) {
        auto const V = points_to_V[point];
        node_V = node_V + (N * V);
      }
    }
    double const m = nodes_to_m[node];
    nodes_to_rho_h[node] = m / node_V;
  };
  lgr::for_each(s.node_sets[material], functor);
}

void interpolate_K(state& s, material_index const material)
{
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_K_h = s.K_h[material].cbegin();
  auto const points_to_K = s.K.begin();
  auto functor = [=] (element_index const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    for (auto const point : elements_to_points[element]) {
      double K = 0.0;
      for (auto const element_node : element_nodes) {
        node_index const node = element_nodes_to_nodes[element_node];
        double const K_h = nodes_to_K_h[node];
        K = lgr::max(K, K_h);
      }
      points_to_K[point] = K;
    }
  };
  lgr::for_each(s.element_sets[material], functor);
}

void interpolate_rho(state& s, material_index const material)
{
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const nodes_to_rho_h = s.rho_h[material].cbegin();
  auto const points_to_rho = s.rho.begin();
  auto const N = 1.0 / double(int(s.nodes_in_element.size()));
  auto functor = [=] (element_index const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    for (auto const point : elements_to_points[element]) {
      double rho = 0.0;
      for (auto const element_node : element_nodes) {
        node_index const node = element_nodes_to_nodes[element_node];
        double const rho_h = nodes_to_rho_h[node];
        rho = rho + rho_h;
      }
      rho = rho * N;
      points_to_rho[point] = rho;
    }
  };
  lgr::for_each(s.element_sets[material], functor);
}

void update_p_h_dot_from_a(input const& in, state& s, material_index const material) {
  update_v_prime(in, s, material);
  update_p_h_W(s, material);
  update_p_h_dot(s, material);
}

void update_e_h_dot_from_a(input const& in, state& s, material_index const material) {
  update_q(in, s, material);
  update_e_h_W(s, material);
  update_e_h_dot(s, material);
}

}
