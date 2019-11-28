//
// Created by himalaya on 10/29/19.
//

#ifndef VMVS_EVOLUTION_H
#define VMVS_EVOLUTION_H

#include "display.h"
#include "init.h"

int Evolve(BoundingBox &box, std::vector<Camera> &all_cams);

dtype Compute_delta(const Grid3d *grid, IdxType i, IdxType j, IdxType k);
#endif //VMVS_EVOLUTION_H
