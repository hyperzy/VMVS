//
// Created by himalaya on 10/24/19.
//

#ifndef VMVS_GRID3D_H
#define VMVS_GRID3D_H

#include "base.h"
#include <list>
#include <queue>

enum NarrowBandStatus {
    ACTIVE,
    LANDMINE,
    BOUNDARY,
    OUTSIDE
};

enum FMM_Status {
    ACCEPT,
    CLOSE,
    FAR,
    OTHER_SIDE
};

enum FrontType {
    // 1 neighbor point on the other side
    TYPE_A,
    // 2 neighbor points on the other side
    TYPE_B1,
    TYPE_B2,
    // 3 neighbor points on the other side
    TYPE_C1,
    TYPE_C2,
    // 4 neighbor points on the other side
    TYPE_D1,
    TYPE_D2,
    // 5 neighbor points on the other side
    TYPE_E,
    // 6 neighbor points on the other side
    TYPE_F
};

enum ExtensionStatus {
    // velocity at the front should be natural
    NATURAL,
    EXTENSION
};

struct NarrowBandExtent {
    IdxType start;
    IdxType end;
};

struct PointProperty {
    int nb_status = NarrowBandStatus::OUTSIDE;
    int fmm_status = FMM_Status::FAR;
    int extension_status;
};

struct IndexPair {
    IdxType i, j, k;
};

// key-value pair, used for heap sort
struct PointKeyVal {
    IdxType i, j, k;
    dtype phi_val;
    bool operator < (const struct PointKeyVal &rhs) const;
};

class Grid3d {
public:
    /**
     * @brief Construct a Grid3 with specified size
     * @param length Size along x axis.
     * @param width Size along y axis.
     * @param height Size along z axis.
     */
    Grid3d(DimUnit length, DimUnit width, DimUnit height);
    std::vector<PointProperty> grid_prop;
    std::vector<dtype> phi;
    // normal velocity
    std::vector<dtype> velocity;
    std::vector<std::vector<std::list<NarrowBandExtent>>> narrow_band;
    std::vector<IndexPair> front;
    // keep record of marching sequence to iterate it again to do extension
    std::vector<IndexPair> marching_sequence;
    std::vector<Point3> coord;
    dtype active_distance;
    dtype landmine_distance;
    dtype boundary_distance;
    const DimUnit length, width, height;
    IdxType band_begin_i, band_end_i, band_begin_j, band_end_j;
    // inline function to compute location
    unsigned long Index(IdxType i, IdxType j, IdxType k);
};


inline unsigned long Grid3d::Index(IdxType i, IdxType j, IdxType k)
{
    return i * this->width * this->height + j * this->height + k;
}
#endif //VMVS_GRID3D_H
