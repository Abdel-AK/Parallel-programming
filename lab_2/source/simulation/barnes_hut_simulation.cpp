#include "simulation/barnes_hut_simulation.h"
#include "simulation/naive_parallel_simulation.h"
#include "physics/gravitation.h"
#include "physics/mechanics.h"

#include <cmath>
#include <queue>

void BarnesHutSimulation::simulate_epochs(Plotter &plotter,
                                          Universe &universe,
                                          std::uint32_t num_epochs,
                                          bool create_intermediate_plots,
                                          std::uint32_t plot_intermediate_epochs) {
  for (int i = 0; i < num_epochs; i++) {
    simulate_epoch(plotter, universe, create_intermediate_plots, plot_intermediate_epochs);
  }
}

void BarnesHutSimulation::simulate_epoch(Plotter &plotter,
                                         Universe &universe,
                                         bool create_intermediate_plots,
                                         std::uint32_t plot_intermediate_epochs) {
  Quadtree quadtree{universe, universe.get_bounding_box(), 0};
  quadtree.calculate_cumulative_masses();
  quadtree.calculate_center_of_mass();
  calculate_forces(universe, quadtree);
  NaiveParallelSimulation::calculate_velocities(universe);
  NaiveParallelSimulation::calculate_positions(universe);
  if (create_intermediate_plots && universe.current_simulation_epoch % plot_intermediate_epochs == 0) {
    plotter.add_bodies_to_image(universe);
    plotter.add_quadtree_to_bitmap(quadtree);
  }
  universe.current_simulation_epoch++;
}

void BarnesHutSimulation::get_relevant_nodes(Universe &universe,
                                             Quadtree &quadtree,
                                             std::vector<QuadtreeNode *> &relevant_nodes,
                                             Vector2d<double> &body_position,
                                             std::int32_t body_index,
                                             double threshold_theta) {
  quadtree.calculate_cumulative_masses();
  quadtree.calculate_center_of_mass();
  std::vector<QuadtreeNode *> nodes_to_inspect{};
  nodes_to_inspect.push_back(quadtree.root);
  while (!nodes_to_inspect.empty()) {
    auto node = nodes_to_inspect.back();
    nodes_to_inspect.pop_back();
    auto dir = node->center_of_mass - body_position;
    const auto theta = node->bounding_box.get_diagonal() / std::sqrt(std::pow(dir[0], 2) + std::pow(dir[1], 2));
    if (node->body_identifier != -1 && node->body_identifier != body_index) {
      relevant_nodes.push_back(node);
    } else if (theta < threshold_theta) {
      relevant_nodes.push_back(node);
    } else {
      nodes_to_inspect.insert(nodes_to_inspect.end(), node->children.begin(), node->children.end());
    }
  }
}

void BarnesHutSimulation::calculate_forces(Universe &universe, Quadtree &quadtree) {
#pragma omp parallel for
  for (int i = 0; i < universe.num_bodies; ++i) {
    std::vector<QuadtreeNode *> relevant_nodes{};
    get_relevant_nodes(universe, quadtree, relevant_nodes, universe.positions[i], i, 0.2);
    Vector2d<double> body_position = universe.positions[i];

    // get body mass
    double body_mass = universe.weights[i];

    Vector2d<double> applied_force_vector;
    for (auto node : relevant_nodes) {
      Vector2d<double> distant_body_position = node->center_of_mass;

      // calculate vector between bodies to get the direction of the gravitational force
      Vector2d<double> direction_vector = distant_body_position - body_position;

      // calculate the distance between the bodies
      double distance = sqrt(pow(direction_vector[0], 2) + pow(direction_vector[1], 2));

      // calculate gravitational force between the bodies
      double force = gravitational_force(body_mass, node->cumulative_mass, distance);

      // create the force vector
      Vector2d<double> force_vector = direction_vector * (force / distance);

      // sum forces applied to body
      applied_force_vector = applied_force_vector + force_vector;
    }
    universe.forces[i] = applied_force_vector;
  }
}
