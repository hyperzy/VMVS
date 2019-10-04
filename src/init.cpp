//
// Created by himalaya on 9/27/19.
//

#include "init.h"

#include <iostream>
using namespace std;
using namespace cv;

vector<Point3> Grid::Compute_back_projection_plane(const Camera &cam, Direction dir)
{
    Mat k_inv = cam.K.inv();
    const Mat &rotation = cam.R;
    const Mat &motion = cam.t;

    vector<Point3> point3d_set;
    point3d_set.reserve(2);
    vector<Point> point2d_set;
    point2d_set.reserve(2);
    switch (dir) {
        case TOP:
            point2d_set.emplace_back(cam.topmost);
            point2d_set.emplace_back(Point(0, cam.topmost.y));
            break;
        case BOTTOM:
            point2d_set.emplace_back(cam.bottommost);
            point2d_set.emplace_back(Point(0, cam.bottommost.y));
            break;
        case LEFT:
            point2d_set.emplace_back(cam.leftmost);
            point2d_set.emplace_back(Point(cam.leftmost.x, 0));
            break;
        case RIGHT:
            point2d_set.emplace_back(cam.rightmost);
            point2d_set.emplace_back(Point(cam.rightmost.x, 0));
            break;
    }
    Mat coord_transform_matrix;
    hconcat(rotation.t(), motion, coord_transform_matrix);
    Mat aux_vec = (Mat_<dtype>(1, 4) << 0, 0, 0, 1);
    coord_transform_matrix.push_back(aux_vec);

    //
    int depth = 50;
    for (const auto &it : point2d_set) {
        // homogeneous coordinate of the point on image
        Vec3 img_coord_homo(it.x, it.y, 1);
        // non-homogeneous coordinates of the back-projection point in camera coordinate system
        Mat bp_coord_camera = depth * k_inv * Mat(img_coord_homo);
        // then transform the coordinate in camera coordinate system into world coordinate system
        bp_coord_camera.push_back(dtype(1));
        Mat bp_coord_world = coord_transform_matrix * bp_coord_camera;
        point3d_set.emplace_back(Point3(bp_coord_world.at<dtype>(0, 0), bp_coord_world.at<dtype>(1, 0), bp_coord_world.at<dtype>(2, 0)));
    }

    return point3d_set;
}

void Grid::Init_grid() {
//    vector<Point3> point3d_set1 = this->Compute_back_projection_plane(this->__all_cams[0], Direction::TOP);
//    vector<Point3> point3d_set2 = this->Compute_back_projection_plane(this->__all_cams[0], Direction::TOP);
//    this->Compute_intersection(this->__all_cams[0], this->__all_cams[4], point3d_set1, point3d_set2);
    this->Determine_bound_coord();
}

Grid::Grid(const std::vector<Camera> &all_cams):__all_cams(all_cams) {

}

Point3 Grid::Compute_intersection(const Camera &cam1, const Camera &cam2,
                                  const std::vector<Point3> &point3d_set1, const std::vector<Point3> &point3d_set2)
{
    const Vec3 &c1 = cam1.t;
    const Vec3 &c2 = cam2.t;
    // construct a normal of the planed represented by two lines
    Vec3 norm1((Vec3(point3d_set1[0]) - c1).cross(Vec3(point3d_set1[1]) - c1));
//    normalize(norm1, norm1);
    Vec3 norm2((Vec3(point3d_set2[0]) - c2).cross(Vec3(point3d_set2[1]) - c2));
//    normalize(norm2, norm2);
// non normalization will not affect the final result. So it was deleted for efficiency.
    // directional vector of the intersected line
    Vec3 line_vec = norm1.cross(norm2);

    // auxiliary line vector, represented the vector in one plane and orthogonal to the intersected line
    Vec3 aux_line_vec = line_vec.cross(norm1);

    // the result comes from algebra
    // \vec(c2 - c1) \cdot norm2 = d * aux_line_vec \cdot norm2
    // res = d * aux_line_vec + c1
    Point3 res = Point3((c2 - c1).dot(norm2) / (aux_line_vec.dot(norm2)) * aux_line_vec + c1);

    return res;
}

void Grid::Determine_bound_coord()
{
    // scale the bound in case of error
    dtype scale_factor = 1.5;

    // __limits store x_min, x_max, y_min, y_max, z_min, z_max respectively;
    this->__limits.resize(6);
    // __bound_coord stores eight vertex of the bounding box
    this->__bound_coord.reserve(8);
    vector<Point3> point3d_set1;
    vector<Point3> point3d_set2;
    Point3 intersected_point;

    // determine top facet
    point3d_set1 = this->Compute_back_projection_plane(this->__all_cams[0], Direction::TOP);
    point3d_set2 = this->Compute_back_projection_plane(this->__all_cams[4], Direction::TOP);
    intersected_point = this->Compute_intersection(this->__all_cams[0], this->__all_cams[4], point3d_set1, point3d_set2);
    dtype z_max = intersected_point.z;

    // determine bottom facet
    point3d_set1 = this->Compute_back_projection_plane(this->__all_cams[0], Direction::BOTTOM);
    point3d_set2 = this->Compute_back_projection_plane(this->__all_cams[4], Direction::BOTTOM);
    intersected_point = this->Compute_intersection(this->__all_cams[0], this->__all_cams[4], point3d_set1, point3d_set2);
    dtype z_min = intersected_point.z;

    // determine left facet
    point3d_set1 = this->Compute_back_projection_plane(this->__all_cams[0], Direction::LEFT);
    point3d_set2 = this->Compute_back_projection_plane(this->__all_cams[4], Direction::RIGHT);
    intersected_point = this->Compute_intersection(this->__all_cams[0], this->__all_cams[4], point3d_set1, point3d_set2);
    dtype x_min = intersected_point.x;

    // determine right facet
    point3d_set1 = this->Compute_back_projection_plane(this->__all_cams[0], Direction::RIGHT);
    point3d_set2 = this->Compute_back_projection_plane(this->__all_cams[4], Direction::LEFT);
    intersected_point = this->Compute_intersection(this->__all_cams[0], this->__all_cams[4], point3d_set1, point3d_set2);
    dtype x_max = intersected_point.x;

    // determine front facet
    point3d_set1 = this->Compute_back_projection_plane(this->__all_cams[2], Direction::LEFT);
    point3d_set2 = this->Compute_back_projection_plane(this->__all_cams[6], Direction::RIGHT);
    intersected_point = this->Compute_intersection(this->__all_cams[2], this->__all_cams[6], point3d_set1, point3d_set2);
    dtype y_min = intersected_point.y;

    // determine back facet
    point3d_set1 = this->Compute_back_projection_plane(this->__all_cams[2], Direction::RIGHT);
    point3d_set2 = this->Compute_back_projection_plane(this->__all_cams[6], Direction::LEFT);
    intersected_point = this->Compute_intersection(this->__all_cams[2], this->__all_cams[6], point3d_set1, point3d_set2);
    dtype y_max = intersected_point.y;

    dtype mid_val = (x_min + x_max) / 2;
    x_min = mid_val - scale_factor * (mid_val - x_min);
    x_max = mid_val + scale_factor * (x_max - mid_val);

    mid_val = (y_min + y_max) / 2;
    y_min = mid_val - scale_factor * (mid_val - y_min);
    y_max = mid_val + scale_factor * (y_max - mid_val);

    mid_val = (z_min + z_max ) / 2;
    z_min = mid_val - scale_factor * (mid_val - z_min);
    z_max = mid_val + scale_factor * (z_max - mid_val);

    this->__limits[1] = x_max;
    this->__limits[0] = x_min;
    this->__limits[3] = y_max;
    this->__limits[2] = y_min;
    this->__limits[5] = z_max;
    this->__limits[4] = z_min;
//    for (const auto &it : this->__limits) {
//        cout << it << endl;
//    }
    this->__bound_coord.emplace_back(Point3(x_min, y_min, z_max));
    this->__bound_coord.emplace_back(Point3(x_max, y_min, z_max));
    this->__bound_coord.emplace_back(Point3(x_max, y_max, z_max));
    this->__bound_coord.emplace_back(Point3(x_min, y_max, z_max));
    this->__bound_coord.emplace_back(Point3(x_min, y_min, z_min));
    this->__bound_coord.emplace_back(Point3(x_max, y_min, z_min));
    this->__bound_coord.emplace_back(Point3(x_max, y_max, z_min));
    this->__bound_coord.emplace_back(Point3(x_min, y_max, z_min));
}

std::vector<dtype> Grid::Get_limits() const{
    return this->__limits;
}

std::vector<Point3> Grid::Get_bound_coord() const {
    return this->__bound_coord;
}

