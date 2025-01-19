#pragma once
#include "structures/vector2d.h"
#include <string>

class BoundingBox {
   public:
    double x_min{};
    double x_max{};
    double y_min{};
    double y_max{};

    std::string get_string();
    double get_diagonal();
    void plotting_sanity_check();
    BoundingBox get_scaled(std::uint32_t scaling_factor);
    BoundingBox() = default;
    explicit BoundingBox(double x1, double x2, double x3, double x4)
        : x_min{x1}, x_max{x2}, y_min{x3}, y_max{x4} {}
    
    bool contains(const Vector2d<double>& pos) const noexcept;
    BoundingBox get_quadrant(std::uint8_t quadrant) const;
};