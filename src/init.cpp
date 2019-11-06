//
// Created by himalaya on 9/27/19.
//

#include "init.h"

#include <iostream>
#include <omp.h>
#include <iomanip>
#include <fstream>
using namespace std;
using namespace cv;

vector<Point3> BoundingBox::Compute_back_projection_plane(const Camera &cam, Direction dir)
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

void BoundingBox::Init() {
//    vector<Point3> point3d_set1 = this->Compute_back_projection_plane(this->__all_cams[0], Direction::TOP);
//    vector<Point3> point3d_set2 = this->Compute_back_projection_plane(this->__all_cams[0], Direction::TOP);
//    this->Compute_intersection(this->__all_cams[0], this->__all_cams[4], point3d_set1, point3d_set2);
    this->Determine_bound_coord();
}

BoundingBox::BoundingBox(const std::vector<Camera> &all_cams,
                        dtype resolution):__all_cams(all_cams), resolution(resolution) {
    visibility_arr.reserve(all_cams.size());
}

Point3 BoundingBox::Compute_intersection(const Camera &cam1, const Camera &cam2,
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

void BoundingBox::Determine_bound_coord()
{
    // scale the bound in case of error
    dtype scale_factor = 1.5;

    // __extents store x_min, x_max, y_min, y_max, z_min, z_max respectively;
    this->__extents.resize(6);
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

    this->__extents[1] = x_max;
    this->__extents[0] = x_min;
    this->__extents[3] = y_max;
    this->__extents[2] = y_min;
    this->__extents[5] = z_max;
    this->__extents[4] = z_min;
//    for (const auto &it : this->__extents) {
//        cout << it << endl;
//    }
    this->__bound_coord.emplace_back(Point3(x_min, y_min, z_min));
    this->__bound_coord.emplace_back(Point3(x_max, y_min, z_min));
    this->__bound_coord.emplace_back(Point3(x_max, y_max, z_min));
    this->__bound_coord.emplace_back(Point3(x_min, y_max, z_min));
    this->__bound_coord.emplace_back(Point3(x_min, y_min, z_max));
    this->__bound_coord.emplace_back(Point3(x_max, y_min, z_max));
    this->__bound_coord.emplace_back(Point3(x_max, y_max, z_max));
    this->__bound_coord.emplace_back(Point3(x_min, y_max, z_max));
}

std::vector<dtype> BoundingBox::Get_extents() const{
    return this->__extents;
}

std::vector<Point3> BoundingBox::Get_bound_coord() const {
    return this->__bound_coord;
}

void Init_sphere_shape(BoundingBox &box, dtype radius)
{
    vector<dtype> extent = box.Get_extents();
    auto nx = (DimUnit)((extent[1] - extent[0]) / box.resolution);
    auto ny = (DimUnit)((extent[3] - extent[2]) / box.resolution);
    auto nz = (DimUnit)((extent[5] - extent[4]) / box.resolution);
//    nx = ny = nz = 33;
    vector<Point3> bound_coord = box.Get_bound_coord();
    Point3 origin = bound_coord[0];
    dtype resolution = box.resolution;
    box.grid3d = new Grid3d(nx, ny, nz);
    box.grid3d->coord.resize(nx * ny * nz);

//    box.grid3d = new Grid3d(20, 20, 20);
    Grid3d *&grid = box.grid3d;
    IdxType center_i = grid->_height / 2;
    IdxType center_j = grid->_width / 2;
    IdxType center_k = grid->_depth / 2;
    auto time_start = omp_get_wtime();
#pragma omp parallel for default(none) shared(center_i, center_j, center_k, grid, radius, origin, resolution)
    for (IdxType i = 0; i < grid->_height; ++i) {
        for (IdxType j = 0; j < grid->_width; ++j) {
            bool flag_interior = false;
            IdxType start = 0, end = 0;
            for (IdxType k = 0; k < grid->_depth; ++k) {
                grid->phi[grid->Index(i, j, k)] = sqrt(pow(i - center_i, 2)
                                                       + pow(j - center_j, 2)
                                                       + pow(k - center_k, 2)) - radius;
                auto &point_coord = grid->coord[grid->Index(i, j, k)];
                point_coord[0] = origin.x + i * resolution;
                point_coord[1] = origin.y + j * resolution;
                point_coord[2] = origin.z + k * resolution;
                if (!flag_interior) {start = k, end = k;}
                else { end = k;}
                auto absolute_val = abs(grid->phi[grid->Index(i, j, k)]);
                auto &nb_status = grid->grid_prop[grid->Index(i, j, k)].nb_status;
                if (absolute_val <= grid->boundary_distance) {
                    flag_interior = true;
                    // build coarse band
                    grid->band_begin_i = i < grid->band_begin_i ? i : grid->band_begin_i;
                    grid->band_end_i = i > grid->band_end_i - 1 ? i : grid->band_end_i;
                    grid->band_begin_j[i] = j < grid->band_begin_j[i] ? j : grid->band_begin_j[i];
                    grid->band_end_j[i] = j > grid->band_end_j[i] - 1 ? j : grid->band_end_j[i];

                    if (absolute_val <= grid->active_distance) { nb_status = NarrowBandStatus::ACTIVE;}
                    else if (absolute_val <= grid->landmine_distance) { nb_status = NarrowBandStatus::LANDMINE;}
                    else {nb_status = NarrowBandStatus::BOUNDARY;}
                }
                else { flag_interior = false;}

                if (start != end && !flag_interior) {
                    grid->narrow_band[i][j].emplace_back(NarrowBandExtent{start, end});
                }
            }
        }
    }
#pragma omp barrier
    cout << omp_get_wtime() -time_start << endl;
    fstream fout("testdata.txt", ios::out);
    for (int z = 0; z < grid->_depth; z++) {
        fout << "z = " << z << endl;
        for (int y = 0; y < grid->_height; y++)
        {fout << scientific << setprecision(3) << setw(5) << setfill('0') << (float)y << " "; }
        fout << endl;
        for (int x = 0; x < grid->_height; x++) {

            for (int y = 0 ; y < grid->_width; y++) {
                fout << scientific << setprecision(3) << setw(5) << setfill('0') << grid->phi[grid->Index(x, y, z)] << " ";
            }
            fout << endl;
        }
        fout << endl;
    }
    fout.close();

    auto &coord = box.grid3d->coord;

    auto new_grid3d = FMM3d(grid, true);
    delete grid;
    box.grid3d = new_grid3d;
    //// initial visibility array
    for (int i = 0; i < box.__all_cams.size(); i++) {
        box.visibility_arr.emplace_back(Visibility(box.__extents, box.__all_cams[i], box.grid3d->coord, resolution,
                                                   box.grid3d->_height, box.grid3d->_width, box.grid3d->_depth));
    }
////#pragma omp parallel for default(none) shared(box)
    for (int i = 0; i < box.visibility_arr.size(); i++) {
        box.visibility_arr[i].Set_phi(box.grid3d->phi);
        box.visibility_arr[i].Calculate_all();
    }
//#pragma omp barrier
}
