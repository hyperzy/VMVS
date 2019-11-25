//
// Created by himalaya on 10/31/19.
//

#ifndef VMVS_VISIBILITY_H
#define VMVS_VISIBILITY_H

#include "base.h"

enum PlaneDirection {
    // sweep from x negative to x positive
    YZ_POS = 0,
    // sweeping from x positive to x negative
    YZ_NEG,
    XZ_POS,
    XZ_NEG,
    XY_POS,
    XY_NEG
};

class Visibility
{
public:
    std::vector<dtype> psi;
    Visibility(const std::vector<dtype> &extent, const Camera &camera, const std::vector<Vec3> &coord,
                dtype resolution, DimUnit height, DimUnit width, DimUnit depth);
    void Set_phi(const std::vector<dtype> &phi);
    unsigned long Index(IdxType i, IdxType j, IdxType k) const;

    /**
     * Determine if point p1 is on the specific side of the plane. We use Vec instead of Point for convenience
     * @param p1
     * @param plane_point A arbitrary point on the plane
     * @param plane_normal A specific normal of the plane. Here we use Point3 instead of Vec3 for convenience
     * @return true if point p1 on the given side
     */
    static bool isPositionHere(const Vec3 &p1, const Vec3 &plane_point, const Vec3& plane_normal);

    /**
     * Determine sweeping direction for given camera
     * @return direction
     */
    int Determine_direction() const;

    /**
     * Determine whether the camera is on discrete grid point or not
     * @return True if the camera is on grid point
     */
    bool isOnGridPoint() const;

    void Calculate_all();

    dtype Calculate_point(IdxType i, IdxType j, IdxType k);

    Vec3 Intersect(const Vec3 &dir_vec, const Vec3 &cur_pt, const Vec3 &plane_pt, int direction);

    dtype Interpolation(const Vec3 &intersect_point, const unsigned long &sur_p1_idx, const unsigned long &sur_p2_idx,
                                                     const unsigned long &sur_p3_idx, const unsigned long &sur_p4_idx, int direction);

    bool isValidRange(IdxType i, IdxType j, IdxType k) const;

private:
    const std::vector<dtype> &extent;
    const Camera &camera;
    const dtype *phi;
    // here we used const pointer instead onf constant reference since there is a swap clause in FMM3d
    Vec3 const *coord;
    const DimUnit height;
    const DimUnit width;
    const DimUnit depth;
    dtype resolution;
    dtype err;
    dtype tolerance;
    std::vector<Vec3> vertices;
    int dir;
    Vec3 cam_coord;
    dtype float_err;
};

inline unsigned long Visibility::Index(IdxType i, IdxType j, IdxType k) const
{
    return i * this->width * this->depth + j * this->depth + k;
}

inline bool Visibility::isValidRange(IdxType i, IdxType j, IdxType k) const
{
    return (i >= 0 && i < this->height && j >= 0 && j < this->width
            && k >= 0 && k < this->depth);
}
#endif //VMVS_VISIBILITY_H
