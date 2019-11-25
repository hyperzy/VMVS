//
// Created by himalaya on 10/24/19.
//

#ifndef VMVS_GRID3D_H
#define VMVS_GRID3D_H

#include "base.h"
#include "velocity_calculator.h"
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

struct IndexSet {
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
    Grid3d(DimUnit height, DimUnit width, DimUnit depth);
    std::vector<PointProperty> grid_prop;
    std::vector<dtype> phi;
//    // normal velocity of Phi* div(\nabla phi / norm(phi))
//    std::vector<dtype> velocity;
//    // normal velocity of innter product of \nabla Phi and \nabla phi / norm(phi)
//    std::vector<dtype> velocity2;
//    // store \nabla Phi to determine the upwind of norm(phi)
//    std::vector<Vec3> d_Phi;
    //// IMPORTANT: from the discussion with Professor, we only need to extend Phi.
    std::vector<dtype> Phi;
    std::vector<std::vector<std::list<NarrowBandExtent>>> narrow_band;
    std::vector<IndexSet> front;
    // keep record of marching sequence to iterate it again to do extension
    std::vector<IndexSet> marching_sequence;
    std::vector<Vec3> coord;
    dtype active_distance;
    dtype landmine_distance;
    dtype boundary_distance;
    const DimUnit _height, _width, _depth;
    IdxType band_begin_i, band_end_i;
    std::vector<IdxType> band_begin_j, band_end_j;
    // we assume number less than 1e-10 is zero.
    // inline function to compute location
    unsigned long Index(IdxType i, IdxType j, IdxType k) const;
    bool isValidRange(IdxType i, IdxType j, IdxType k) const;

    /**
     * For now LANDMINE is considered, but actually it seems that we dont need LANDMINE since these will never
     * when evolving.
     * This data structure is arranged as a list consisting of one or more nodes. For each node, there is a
     * 'start' and 'end' which is the index of a band interval, where 'end' is one more step of the last
     * element of each band interval.
     */
    void Build_band();

    /**
     *  Construct a coarse narrow band (or say a roughly region that is active).
     *  i, j and k are the current point index, I want to determine the range (from minimum to maximum column index) of
     *  each row.
     * @param i
     * @param j
     * @param k
     */
    void Build_coarse_band(IdxType i, IdxType j, IdxType k);

    /**
     *  Determine if the front is around this point( bottom, top, left, right, the front, back)
     * @param i Height index
     * @param j Width index
     * @param k Depth index
     * @param front_dir Direction where front lies
     * @return True if the front is near the grid point.
     */
    bool isFrontHere(IdxType i, IdxType j, IdxType k, std::vector<cv::Vec3i> &front_dir) const;



    /**
     *  Actually, maybe this method need to be taken apart for positive and negative since positive dont need to
     *  log the ACCEPT index for negating numbers.
     * @param close_set
     * @param inside True for negative value (inside the surface). False otherwise.
     */
    void Marching(std::priority_queue<PointKeyVal> &close_set, bool inside);



    void Extend_velocity(PhiCalculator *velocity_calculator);
    void Update_velocity(PhiCalculator *velocity_calculator);

    Grid3d *Reinitialize(PhiCalculator *velocity_calculator);
};


inline unsigned long Grid3d::Index(IdxType i, IdxType j, IdxType k) const
{
    return i * this->_width * this->_depth + j * this->_depth + k;
}

/**
 * @param initial_grid
 * @param reinit True for re-initialization. False for not reinitialization
 * @param velocity_calculator The calculator for computing Discrepency Measurement Phi
 * @return
 */
Grid3d* FMM3d(Grid3d *initial_grid, bool reinit, PhiCalculator *velocity_calculator);

/**
 * The implementation of this function refers to original paper.
 * @param old_grid
 * @param new_grid
 * @param i
 * @param j
 * @param k
 * @param front_dir The directions where the front lies
 */
void Determine_front_property(Grid3d *old_grid, Grid3d *new_grid,
                              IdxType i, IdxType j, IdxType k,
                              std::vector<cv::Vec3i> &front_dir);

dtype Determine_velocity_negative(const std::vector<dtype> &phi, IdxType i, IdxType j, IdxType k, const std::vector<cv::Vec3i> &front_dir);

#endif //VMVS_GRID3D_H
