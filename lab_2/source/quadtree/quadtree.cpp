#include "quadtree.h"

#include "quadtreeNode.h"
#include <set>
#include <algorithm>
#include <stdexcept>
#include <omp.h>

Quadtree::Quadtree(Universe &universe, BoundingBox bounding_box, std::int8_t construct_mode) {
  std::vector<std::int32_t> body_indices{};
  for (std::int32_t i = 0; i < universe.num_bodies; ++i) {
    if (bounding_box.contains(universe.positions[i])) {
      body_indices.push_back(i);
    }
  }
  root = new QuadtreeNode(bounding_box);
  switch (construct_mode) {
    case 0:
      root->children = construct(universe, bounding_box, body_indices);
      break;
    case 1:
#pragma omp parallel
    {
#pragma omp single
      root->children = construct_task(universe, bounding_box, body_indices);
    }
    break;
    case 2:
      root->children = construct_task_with_cutoff(universe, bounding_box, body_indices);
      break;
    default:
      delete root;
      break;
  }
}

Quadtree::~Quadtree() {
  delete root;
}

void Quadtree::calculate_cumulative_masses() {
  root->calculate_node_cumulative_mass();
}

void Quadtree::calculate_center_of_mass() {
  root->calculate_node_center_of_mass();
}

std::vector<QuadtreeNode *> Quadtree::construct(Universe &universe,
                                                BoundingBox BB,
                                                std::vector<std::int32_t> body_indices) {
  std::vector<QuadtreeNode *> children;
  for (int i = 0; i < 4; ++i) {
    auto bb = BB.get_quadrant(i);
    std::vector<std::int32_t> bb_indices{};
    std::ranges::copy_if(body_indices,
                         std::back_inserter(bb_indices),
                         [&](auto index) {
                           return bb.contains(universe.positions[index]);
                         });
    if (!bb_indices.empty()) {
      children.push_back(new QuadtreeNode(bb));
      if (bb_indices.size() == 1) {
        children.back()->body_identifier = bb_indices[0];
        children.back()->center_of_mass = universe.positions[bb_indices[0]];
        children.back()->cumulative_mass = universe.weights[bb_indices[0]];
      } else {
        children.back()->children = construct(universe, bb, bb_indices);
      }
    }
  }
  return children;
}

std::vector<QuadtreeNode *> Quadtree::construct_task(Universe &universe,
                                                     BoundingBox BB,
                                                     std::vector<std::int32_t> body_indices) {
  std::vector<QuadtreeNode *> children;
  for (int i = 0; i < 4; ++i) {
    auto bb = BB.get_quadrant(i);
    std::vector<std::int32_t> bb_indices{};
    std::ranges::copy_if(body_indices,
                         std::back_inserter(bb_indices),
                         [&](auto index) {
                           return bb.contains(universe.positions[index]);
                         });
    if (!bb_indices.empty()) {
      children.push_back(new QuadtreeNode(bb));
      if (bb_indices.size() == 1) {
        children.back()->body_identifier = bb_indices[0];
        children.back()->center_of_mass = universe.positions[bb_indices[0]];
        children.back()->cumulative_mass = universe.weights[bb_indices[0]];
      } else {
#pragma omp task
        {
          children.back()->children = construct_task(universe, bb, bb_indices);
        }
      }
    }
  }
  return children;
}

std::vector<QuadtreeNode *> Quadtree::construct_task_with_cutoff(Universe &universe,
                                                                 BoundingBox &BB,
                                                                 std::vector<std::int32_t> &body_indices) {
  std::vector<QuadtreeNode *> children;
  for (int i = 0; i < 4; ++i) {
    auto bb = BB.get_quadrant(i);
    std::vector<std::int32_t> bb_indices{};
    std::ranges::copy_if(body_indices,
                         std::back_inserter(bb_indices),
                         [&](auto index) {
                           return bb.contains(universe.positions[index]);
                         });
    if (!bb_indices.empty()) {
      children.push_back(new QuadtreeNode(bb));
      if (bb_indices.size() == 1) {
        children.back()->body_identifier = bb_indices[0];
        children.back()->center_of_mass = universe.positions[bb_indices[0]];
        children.back()->cumulative_mass = universe.weights[bb_indices[0]];
      } else if (bb_indices.size() > 100){
#pragma omp task
        {
          children.back()->children = construct_task_with_cutoff(universe, bb, bb_indices);
        }
      }else {
        children.back()->children = construct_task_with_cutoff(universe, bb, bb_indices);
      }
    }
  }
  return children;
}

std::vector<BoundingBox> Quadtree::get_bounding_boxes(QuadtreeNode *qtn) {
  // traverse quadtree and collect bounding boxes
  std::vector<BoundingBox> result;
  // collect bounding boxes from children
  for (auto child : qtn->children) {
    for (auto bb : get_bounding_boxes(child)) {
      result.push_back(bb);
    }
  }
  result.push_back(qtn->bounding_box);
  return result;
}
