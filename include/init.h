//
// Created by himalaya on 9/27/19.
//

#ifndef VMVS_INIT_H
#define VMVS_INIT_H
/*
 * Note: a rectangle cube will be used for the bounding box
 */

#include "base.h"
#include "grid3d.h"
#include "visibility.h"
#include "velocity_calculator.h"

enum Direction {
    TOP,
    BOTTOM,
    LEFT,
    RIGHT
};


class BoundingBox {
private:
    // [xmin, xmax, ymin, ymax, zmin, zmax]
    //
    std::vector<Point3> __bound_coord;


    /** Computing back projection plane represented by two lines (since two lines compose a plane).
    *
    * @cam_param: camera parameters comprised of intrinsic and extrinsic parameters
    * @seg_img: rough segmentation image
    * @dir: determine the direction of the relative plane location to object.
    *
    * return a vector containing two points. Each line is represented by the respective point and focal point.
    */
    static std::vector<Point3> Compute_back_projection_plane(const Camera &cam, Direction dir);

    /** Compute the intersection line of two plane and return one point on the line.
     *
     * @cam1: camera parameters of camera 1
     * @cam2: camera parameters of camera 2
     * @point3d_set_arr: a vector consisting of two set of two points on two different lines representing a plane.
     *
     * return a 3d point on the intersection line.
     */
    static Point3 Compute_intersection(const Camera &cam1, const Camera &cam2, const std::vector<Point3> &point3d_set1, const std::vector<Point3> &point3d_set2);

    /**  Compute the bounding box corner coordinates and the boundary(surface) coordinates
     *
     * @param cam_arr Camera Parameters array
     * @param seg_img_arr Rough segmentation image array
     * @param extents Boundary(Surface) coordinates array
     * @param bound_coord Bounding box vertex coordinates
     */
    void Determine_bound_coord();

public:

    explicit BoundingBox(const std::vector<Camera> &all_cams, dtype resolution);
    ~BoundingBox();
    Grid3d *grid3d;
    const dtype resolution;
    std::vector<Visibility> visibility_arr;
    const std::vector<Camera> &__all_cams;
    std::vector<dtype> __extents;

    /**
     * @brief Determine the bounding box coordinates and
     */
    void Init();
    std::vector<dtype> Get_extents() const;
    std::vector<Point3> Get_bound_coord() const;
    PhiCalculator *velocity_calculator;
};

// radius is not in use
void Init_sphere_shape(BoundingBox &grid, dtype radius);







#endif //VMVS_INIT_H
