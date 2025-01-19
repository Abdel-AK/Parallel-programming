#include "simulation/naive_sequential_simulation.h"

#include <cmath>

#include "naive_sequential_simulation.h"
#include "physics/gravitation.h"
#include "physics/mechanics.h"
#include "simulation/constants.h"

void NaiveSequentialSimulation::simulate_epochs(Plotter& plotter, Universe& universe, std::uint32_t num_epochs, bool create_intermediate_plots,
                                                std::uint32_t plot_intermediate_epochs) {
    for (std::uint32_t i = 0; i < num_epochs; ++i) {
        simulate_epoch(plotter, universe, create_intermediate_plots, plot_intermediate_epochs);
    }
}

void NaiveSequentialSimulation::simulate_epoch(Plotter& plotter, Universe& universe, bool create_intermediate_plots,
                                               std::uint32_t plot_intermediate_epochs) {
    calculate_forces(universe);
    calculate_velocities(universe);
    calculate_positions(universe);
    universe.current_simulation_epoch++;
    if (create_intermediate_plots && universe.current_simulation_epoch % plot_intermediate_epochs == 0) {
        plotter.add_bodies_to_image(universe);
        plotter.highlight_position(universe.positions[0], 255, 0, 0);
        plotter.write_and_clear();
    }
}

void NaiveSequentialSimulation::calculate_forces(Universe& universe) {
    auto n = universe.num_bodies;
    for (std::uint32_t i = 0u; i < n; ++i) {
        Vector2d<double> sum = Vector2d{0., 0.};

        for (std::uint32_t j = 0; j < i; ++j) {
            auto vec = universe.positions[j] - universe.positions[i];
            auto dist = std::sqrt(std::pow(vec[0], 2) + std::pow(vec[1], 2));
            auto force = gravitational_force(universe.weights[i], universe.weights[j], dist);
            auto normed = vec / dist;
            auto force_dir = normed * force;
            sum += force_dir;
        }
        for (std::uint32_t j = i + 1; j < n; ++j) {
            auto vec = universe.positions[j] - universe.positions[i];
            auto dist = std::sqrt(std::pow(vec[0], 2) + std::pow(vec[1], 2));
            auto force = gravitational_force(universe.weights[i], universe.weights[j], dist);
            auto normed = vec / dist;
            auto force_dir = normed * force;
            sum += force_dir;
        }
        universe.forces[i] = sum;
    }
}

void NaiveSequentialSimulation::calculate_velocities(Universe& universe) {
    for (std::uint32_t i = 0; i < universe.num_bodies; i++) {
        universe.velocities[i] =
            calculate_velocity(universe.velocities[i], calculate_acceleration(universe.forces[i], universe.weights[i]), epoch_in_seconds);
    }
}

void NaiveSequentialSimulation::calculate_positions(Universe& universe) {
    for (std::uint32_t i = 0; i < universe.num_bodies; ++i) {
        auto s = universe.velocities[i] * epoch_in_seconds;
        auto pos = universe.positions[i] + s;
        universe.positions[i] = pos;
    }
}
