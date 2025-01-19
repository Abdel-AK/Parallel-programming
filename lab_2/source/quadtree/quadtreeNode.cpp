#include "quadtreeNode.h"

#include <iostream>

double QuadtreeNode::calculate_node_cumulative_mass() {
  if (cumulative_mass_ready) {
    return cumulative_mass;
  }
  for (const auto child : children) {
    cumulative_mass += child->calculate_node_cumulative_mass();
  }
  cumulative_mass_ready = true;
  return cumulative_mass;
}

QuadtreeNode::QuadtreeNode(BoundingBox arg_bounding_box) : bounding_box{arg_bounding_box}, cumulative_mass{0.0},center_of_mass{0.0,0.0} {
}

QuadtreeNode::~QuadtreeNode() {
  for (auto &node : children) {
    delete node;
  }
  children.clear();
}

Vector2d<double> QuadtreeNode::calculate_node_center_of_mass() {
  if (center_of_mass_ready) {
    return center_of_mass;
  }
  if (body_identifier == -1) {
    for (const auto child : children) {
      center_of_mass = center_of_mass + child->calculate_node_center_of_mass() * child->calculate_node_cumulative_mass();
    }
    center_of_mass = center_of_mass / calculate_node_cumulative_mass();
  }
  center_of_mass_ready = true;
  return center_of_mass;
}
