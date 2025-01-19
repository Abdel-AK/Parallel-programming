#pragma once
#include "structures/vector2d.h"

template<typename T>
static Vector2d<T> calculate_acceleration(const Vector2d<T>& force, const T& m){
    return force / m;
}

template<typename T>
static Vector2d<T> calculate_velocity(const Vector2d<T>& v0, const Vector2d<T>& a, const T& t){
    return v0 + a * t;
}