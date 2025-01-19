#pragma once

#include <cstdint>
#include <stdexcept>

template <typename T>
class Vector2d {
   public:
    using value_type = T;

    Vector2d() = default;

    explicit Vector2d(const T& x, const T& y) : x{x}, y{y} {}

    T operator[](std::size_t pos) const {
        if (pos > 1) {
            throw std::out_of_range{"Range 0-1 allowed, got: " + pos};
        } else if (pos == 1) {
            return y;
        }
        return x;
    }

    void set(const T& x, const T& y) {
        this->x = x;
        this->y = y;
    }

    Vector2d<T>& operator+=(const Vector2d<T>& rhs) {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }
    friend Vector2d<T> operator+(Vector2d<T> lhs, const Vector2d<T>& rhs) {
        lhs += rhs;
        return lhs;
    }
    Vector2d<T>& operator-=(const Vector2d<T>& rhs) {
        x -= rhs.x;
        y -= rhs.y;
        return *this;
    }
    friend Vector2d<T> operator-(Vector2d<T> lhs, const Vector2d<T>& rhs) {
        lhs -= rhs;
        return lhs;
    }

    Vector2d<T>& operator*=(const T& rhs) {
        x *= rhs;
        y *= rhs;
        return *this;
    }
    friend Vector2d<T> operator*(Vector2d<T> lhs, const T& rhs) {
        lhs *= rhs;
        return lhs;
    }

    Vector2d<T>& operator/=(const T& rhs) {
        
        x /= rhs;
        y /= rhs;
        return *this;
    }
    friend Vector2d<T> operator/(Vector2d<T> lhs, const T& rhs) {
        lhs /= rhs;
        return lhs;
    }

    auto operator<=>(const Vector2d<T>&) const noexcept = default;

   private:
    T x;
    T y;
};