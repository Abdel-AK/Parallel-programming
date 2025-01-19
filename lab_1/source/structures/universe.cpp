#include "structures/universe.h"

#include <limits>

BoundingBox Universe::get_bounding_box() const noexcept {
    // STL way of doing this, does go through the vector twice, so potentially slower
    // const auto& [x_min, x_max] = std::ranges::minmax_element(positions, std::ranges::less{}, [](const auto& el) { return el[0]; });
    // const auto& [y_min, y_max] = std::ranges::minmax_element(positions, std::ranges::less{}, [](const auto& el) { return el[1]; });

    double x_min{std::numeric_limits<double>::max()}, x_max{std::numeric_limits<double>::min()}, y_min{std::numeric_limits<double>::max()},
        y_max{std::numeric_limits<double>::min()};
    for (const auto& vec : positions) {
        if (x_min > vec[0]) {
            x_min = vec[0];
        }
        if (x_max < vec[0]) {
            x_max = vec[0];
        }
        if (y_min > vec[1]) {
            y_min = vec[1];
        }
        if (y_max > vec[1]) {
            y_max = vec[1];
        }
    }
    return BoundingBox{x_min, x_max, y_min, y_max};
}
