#include "plotting/plotter.h"

#include <exception>

#include "io/image_parser.h"



void Plotter::highlight_position(Vector2d<double> position, std::uint8_t red, std::uint8_t green, std::uint8_t blue) {
    if (!plot_bounding_box.contains(position)) {
        return;
    }
    auto x_scale = (plot_width - 1) / std::abs(plot_bounding_box.x_max - plot_bounding_box.x_min);
    auto y_scale = (plot_height - 1) / std::abs(plot_bounding_box.y_max - plot_bounding_box.y_min);
    auto x_pos = (position[0] - plot_bounding_box.x_min) * x_scale;
    auto y_pos = (position[1] - plot_bounding_box.y_min) * y_scale;
    for (std::uint32_t i = 0; i < plot_height; ++i) {
        mark_pixel(x_pos, i, red, green, blue);
        //image.set_pixel(i, x_pos, BitmapImage::BitmapPixel{red, green, blue});
    }
    for (std::uint32_t i = 0; i < plot_width; ++i) {
        mark_pixel(i, y_pos, red, green, blue);
        //image.set_pixel(y_pos, i, BitmapImage::BitmapPixel{red, green, blue});
    }
}

void Plotter::write_and_clear() {
    // create plot serial number string
    std::string serial_number_string = std::to_string(image_serial_number);
    while (serial_number_string.length() < 9) {
        serial_number_string = "0" + serial_number_string;
    }

    std::string file_name = filename_prefix + "_" + serial_number_string + ".bmp";
    ImageParser::write_bitmap(output_folder_path / file_name, image);
    clear_image();
    image_serial_number += 1;
}

BitmapImage::BitmapPixel Plotter::get_pixel(std::uint32_t x, std::uint32_t y) { return image.get_pixel(y, x); }

void Plotter::mark_position(Vector2d<double> position, std::uint8_t red, std::uint8_t green, std::uint8_t blue) {
    if (!plot_bounding_box.contains(position)) {
        return;
    }
    auto x_scale = (plot_width - 1) / std::abs(plot_bounding_box.x_max - plot_bounding_box.x_min);
    auto y_scale = (plot_height - 1) / std::abs(plot_bounding_box.y_max - plot_bounding_box.y_min);
    auto x_pos = (position[0] - plot_bounding_box.x_min) * x_scale;
    auto y_pos = (position[1] - plot_bounding_box.y_min) * y_scale;
    mark_pixel(x_pos, y_pos, red, green, blue);
    // image.set_pixel(y_pos, x_pos, BitmapImage::BitmapPixel{red, green, blue});
}

void Plotter::mark_pixel(std::uint32_t x, std::uint32_t y, std::uint8_t red, std::uint8_t green, std::uint8_t blue) {
    if (x >= plot_width || y >= plot_height) {
        throw std::exception{};
    }

    image.set_pixel(y, x, BitmapImage::BitmapPixel{red, green, blue});
}
