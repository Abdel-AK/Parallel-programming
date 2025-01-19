#pragma once
#include <fstream>

#include "structures/universe.h"

static void save_universe(const std::filesystem::path& path, const Universe& universe) {
    std::ofstream file{path};

    file << "###Bodies\n";
    file << universe.num_bodies << "\n";

    file << "###Positions\n";
    for (const auto& pos : universe.positions) {
        file << std::setprecision(6) << pos[0] << ' ' << pos[1] << '\n';
    }

    file << "###Weights\n";
    for (const auto& mass : universe.weights) {
        file << std::setprecision(6) << mass << '\n';
    }

    file << "###Velocities\n";
    for (const auto& vel : universe.velocities) {
        file << std::setprecision(6) << vel[0] << ' ' << vel[1] << '\n';
    }

    file << "###Forces\n";
    for (const auto& force : universe.forces) {
        file << std::setprecision(6) << force[0] << ' ' << force[1] << '\n';
    }
}