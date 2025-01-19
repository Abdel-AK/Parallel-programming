#include "plotting/plotter.h"

void Plotter::add_bodies_to_image(Universe& universe) {
    for(const auto& pos : universe.positions){
        if(plot_bounding_box.contains(pos)){
            mark_position(pos, 255, 255, 255);
        }
    }
}