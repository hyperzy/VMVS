//
// Created by himalaya on 10/6/19.
//

#ifndef VMVS_GRID2D_H
#define VMVS_GRID2D_H

//// This version changes the logistics that
//// 1. swap the phi_val since we want to keep phi_val before landmine is hit
//// 2. keep narrow band data structure until landmine is hit
//// 3. separate velocity from PointProperty


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
    // only 1 point on the other side
    TYPE_A = 1,
    // only two adjacent points on the other side
    TYPE_B,
    // three points on the other side
    TYPE_C,
    // two non-adjacent points on the other side
    TYPE_D,
    // four points on the other side
    TYPE_E
};

enum ExtensionStatus {
    // velocity at front should be natural
    NATURAL,
    EXTENSION
};

struct NarrowBandBound {
    unsigned short start;
    unsigned short end;
};

struct PointProperty {
    int nb_status = NarrowBandStatus::OUTSIDE;
    int fmm_status = FMM_Status::FAR;
    int extension_status;
    PointProperty& operator=(const PointProperty &rhs);

};

struct IndexPair {
    unsigned short i;
    unsigned short j;
};

struct PointKeyVal {
    unsigned short i;
    unsigned short j;
    dtype phi_val;
    bool operator < (const struct PointKeyVal &rhs) const;
};

class Grid2d {
public:
    Grid2d();
    Grid2d(unsigned short length, unsigned short height);
    std::vector<PointProperty> grid;
    std::vector<dtype> phi;
    // normal velocity
    std::vector<dtype> velocity;
    std::vector<std::list<NarrowBandBound>> narrow_band;
    std::vector<IndexPair> front;
    std::vector<IndexPair> marching_sequence;
    std::vector<Point3> coord;
    dtype active_bandwidth;
    dtype landmine_distance;
    dtype boundary_distance;
    const unsigned short height, width;
    unsigned short band_begin_i, band_end_i;
    inline unsigned long Index(unsigned short i, unsigned short j);
    void FMM_init();
    void Approx_front();
    /**
     *  Determine if the front is around this point( bottom, top, left, right)
     * @param i Row index
     * @param j Column index
     * @return
     */
    unsigned short isFrontHere(unsigned long i, unsigned long j);

    /**
     * For now LANDMINE is considered, but actually it seems that we dont need LANDMINE since these will never
     * when evolving.
     */
    void Build_band();
    bool isValidRange(unsigned short i, unsigned short j);

    /**
     *  Actually, maybe this method need to be taken apart for positive and negative since positive dont need to
     *  log the ACCEPT index for negating numbers.
     * @param close_set
     * @param active_bandwidth
     * @param landmine_distance
     * @param inside
     */
    void Marching(std::priority_queue<PointKeyVal> &close_set, bool inside);

    /**
     *  Construct a coarse narrow band (or say a roughly region that is active).
     *  i and j are the current point index, I want to determine the range (from minimum to maximum column index) of
     *  each row.
     * @param i
     * @param j
     */
    void Build_coarse_band(unsigned short i, unsigned short j);

    void Extend_velocity();
    void Extend_velocity(int fmm_status, unsigned short i, unsigned short j);
    void Update_velocity();
    Grid2d* Reinitialize();
};

void Zero_val_handler(Grid2d &old_grid, Grid2d &new_grid,
                        unsigned short i, unsigned short j,
                        std::priority_queue<PointKeyVal> &close_pq_pos,
                        std::priority_queue<PointKeyVal> &close_pq_neg);

/**
 *
 * @param initial_grid
 * @param reinit True for re-initialization. False for not reinitialization
 * @return
 */
Grid2d* FMM2d(Grid2d *initial_grid, bool reinit);

/**
 * The implementation of this function refers to original paper.
 * @param old_grid
 * @param new_grid
 * @param i
 * @param j
 * @param sign_changed
 */
void Determine_front_property(Grid2d *old_grid, Grid2d *new_grid,
                              unsigned short i, unsigned short j, unsigned short sign_changed);

void Evolve(Grid2d *old_grid);
#endif //VMVS_GRID2D_H
