#include "simulation/barnes_hut_simulation_with_collisions.h"
#include "simulation/barnes_hut_simulation.h"
#include "simulation/naive_parallel_simulation.h"
#include <omp.h>
#include <algorithm>

void BarnesHutSimulationWithCollisions::simulate_epochs(Plotter &plotter,
                                                        Universe &universe,
                                                        std::uint32_t num_epochs,
                                                        bool create_intermediate_plots,
                                                        std::uint32_t plot_intermediate_epochs) {
  for (int i = 0; i < num_epochs; i++) {
    simulate_epoch(plotter, universe, create_intermediate_plots, plot_intermediate_epochs);
  }
}

void BarnesHutSimulationWithCollisions::simulate_epoch(Plotter &plotter,
                                                       Universe &universe,
                                                       bool create_intermediate_plots,
                                                       std::uint32_t plot_intermediate_epochs) {
  Quadtree quadtree{universe, universe.get_bounding_box(), 0};
  quadtree.calculate_cumulative_masses();
  quadtree.calculate_center_of_mass();
  calculate_forces(universe, quadtree);
  NaiveParallelSimulation::calculate_velocities(universe);
  NaiveParallelSimulation::calculate_positions(universe);
  find_collisions(universe);
  if (create_intermediate_plots && universe.current_simulation_epoch % plot_intermediate_epochs == 0) {
    plotter.add_bodies_to_image(universe);
    plotter.add_quadtree_to_bitmap(quadtree);
  }
  universe.current_simulation_epoch++;
}

void BarnesHutSimulationWithCollisions::find_collisions(Universe &universe) {
  std::vector<std::pair<int32_t, int32_t> > pairs{};
  for (int i = 0; i < universe.num_bodies; ++i) {
    for (int j = i + 1; j < universe.num_bodies; ++j) {
      auto dist = universe.positions[i] - universe.positions[j];
      if (std::sqrt(std::pow(dist[0], 2) + std::pow(dist[1], 2)) < 100'000'000'000) {
        pairs.emplace_back(i, j);
      }
    }
  }
  while (!pairs.empty()) {
    std::ranges::sort(pairs,
                      [](const auto first, const auto second) {
                        if (auto biggest = std::max(std::max(first.first, second.first),
                                                    std::max(second.second, first.second)); biggest != first.first &&
                          biggest != first.second)
                          return false;
                        return true;
                      });
    auto [p_first, p_second] = pairs.front();
    auto momentum = universe.velocities[p_first] * universe.weights[p_first] + universe.velocities[p_second] *
        universe.weights[p_second];
    int32_t removable;
    if (universe.weights[p_first] <= universe.weights[p_second]) {
      universe.weights[p_second] = universe.weights[p_first] + universe.weights[p_second];
      universe.weights[p_first] = 0;
      removable = p_first;
      auto vel = momentum / universe.weights[p_second];
      universe.velocities[p_second] = vel;
    } else {
      universe.weights[p_first] = universe.weights[p_second] + universe.weights[p_first];
      universe.weights[p_second] = 0;
      removable = p_second;
      auto vel = momentum / universe.weights[p_first];
      universe.velocities[p_first] = vel;
    }
    auto [first, last] = std::ranges::remove_if(pairs,
                                                [&](const auto test) {
                                                  return test.first == removable || test.second == removable;
                                                });
    pairs.erase(first, last);
    universe.forces.erase(universe.forces.begin() + removable);
    universe.positions.erase(universe.positions.begin() + removable);
    universe.velocities.erase(universe.velocities.begin() + removable);
    universe.weights.erase(universe.weights.begin() + removable);
    universe.num_bodies--;
    for (auto &[fst, snd] : pairs) {
      if (fst > removable) {
        fst--;
      }
      if (snd > removable) {
        snd--;
      }
    }
  }
}

void BarnesHutSimulationWithCollisions::find_collisions_parallel(Universe &universe) {
  std::vector<std::pair<int32_t, int32_t>> pairs{};
#pragma omp parallel for collapse(2)
  for (int i = 0; i < universe.num_bodies; ++i) {
    for (int j = i + 1; j < universe.num_bodies; ++j) {
      auto dist = universe.positions[i] - universe.positions[j];
      if (std::sqrt(std::pow(dist[0], 2) + std::pow(dist[1], 2)) < 100'000'000'000) {
        pairs.emplace_back(i, j);
      }
    }
  }
  while (!pairs.empty()) {
    std::ranges::sort(pairs,
                      [](const auto first, const auto second) {
                        if (auto biggest = std::max(std::max(first.first, second.first),
                                                    std::max(second.second, first.second)); biggest != first.first &&
                          biggest != first.second)
                          return false;
                        return true;
                      });
    auto [p_first, p_second] = pairs.front();
    auto momentum = universe.velocities[p_first] * universe.weights[p_first] + universe.velocities[p_second] *
        universe.weights[p_second];
    int32_t removable;
    if (universe.weights[p_first] <= universe.weights[p_second]) {
      universe.weights[p_second] = universe.weights[p_first] + universe.weights[p_second];
      universe.weights[p_first] = 0;
      removable = p_first;
      auto vel = momentum / universe.weights[p_second];
      universe.velocities[p_second] = vel;
    } else {
      universe.weights[p_first] = universe.weights[p_second] + universe.weights[p_first];
      universe.weights[p_second] = 0;
      removable = p_second;
      auto vel = momentum / universe.weights[p_first];
      universe.velocities[p_first] = vel;
    }
    auto [first, last] = std::ranges::remove_if(pairs,
                                                [&](const auto test) {
                                                  return test.first == removable || test.second == removable;
                                                });
    pairs.erase(first, last);
    universe.forces.erase(universe.forces.begin() + removable);
    universe.positions.erase(universe.positions.begin() + removable);
    universe.velocities.erase(universe.velocities.begin() + removable);
    universe.weights.erase(universe.weights.begin() + removable);
    universe.num_bodies--;
    for (auto &[fst, snd] : pairs) {
      if (fst > removable) {
        fst--;
      }
      if (snd > removable) {
        snd--;
      }
    }
  }
}
