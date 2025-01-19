#pragma once
#include <cstdint>
#include <vector>
#include "structures/vector2d.h"
#include "structures/bounding_box.h"

class Universe {
    public:
    std::uint32_t num_bodies{};
    std::uint32_t current_simulation_epoch{};
    std::vector<double> weights{};
    std::vector<Vector2d<double>> forces{};
    std::vector<Vector2d<double>> velocities{};
    std::vector<Vector2d<double>> positions{};

    BoundingBox get_bounding_box() const noexcept;
};