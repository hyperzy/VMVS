//
// Created by himalaya on 10/3/19.
//

#ifndef VMVS_DISPLAY_H
#define VMVS_DISPLAY_H

#include "base.h"
#include "init.h"

#define USE_NEW 0
#if USE_NEW
void Show_3D(std::vector<Camera> &all_cams, BoundingBox &box);
#else
void Show_3D(const std::vector<Camera> &all_cams, const BoundingBox &box);
#endif

#endif //VMVS_DISPLAY_H
