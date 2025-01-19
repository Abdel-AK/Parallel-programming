#pragma once

#include <cmath>

static constexpr double gravitational_constant = 6.67430e-11;
static constexpr double gravitational_force(const double m1, const double m2, const double d){
    return gravitational_constant * m1 * m2 / (d*d);
}