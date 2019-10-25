//
// Created by himalaya on 10/24/19.
//

#include "grid3d.h"


Grid3d::Grid3d(DimUnit length, DimUnit width, DimUnit height):
        length(length), width(width), height(height),
        active_distance(4.1), landmine_distance(5.1), boundary_distance(6.1)
{
    unsigned long total_num = length * width * height;
    this->grid_prop.resize(total_num);
    this->phi.resize(total_num);
    this->velocity.resize(total_num);
    this->narrow_band.resize(length);
    for (auto iter : this->narrow_band) {iter.resize(width);}
    this->band_begin_i = length;
    this->band_end_i = 0;
    this->band_begin_j = width;
    this->band_end_j = 0;
}

