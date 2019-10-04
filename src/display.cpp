//
// Created by himalaya on 10/3/19.
//

#include "display.h"

#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);

#include <vtkActor.h>
#include <vtkCylinderSource.h>
#include <vtkNamedColors.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>
#include <vtkCubeSource.h>
#include <vtkPoints.h>
#include <vtkAssembly.h>
#include <vtkLineSource.h>

using namespace std;

vtkSmartPointer<vtkActor> Construct_lineActor(const Point3 &p1, const Point3 &p2)
{
    vtkSmartPointer<vtkNamedColors> colors = vtkSmartPointer<vtkNamedColors>::New();
    vtkSmartPointer<vtkLineSource> line = vtkSmartPointer<vtkLineSource>::New();
    dtype p1_arr[3] = {p1.x, p1.y, p1.z};
    dtype p2_arr[3] = {p2.x, p2.y, p2.z};
    line->SetPoint1(p1_arr);
    line->SetPoint2(p2_arr);
    line->Update();
    vtkSmartPointer<vtkPolyDataMapper> line_mapper= vtkSmartPointer<vtkPolyDataMapper>::New();
    line_mapper->SetInputData(line->GetOutput());
    vtkSmartPointer<vtkActor> line_actor = vtkSmartPointer<vtkActor>::New();
    line_actor->SetMapper(line_mapper);
    line_actor->GetProperty()->SetLineWidth(1.5);
    line_actor->GetProperty()->SetColor(colors->GetColor3d("Black").GetData());
    return line_actor;
}

void Show_3D(const vector<Camera> &all_cams, const Grid &grid)
{
    vtkSmartPointer<vtkNamedColors> colors = vtkSmartPointer<vtkNamedColors>::New();
    vtkSmartPointer<vtkRenderer> ren = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renw = vtkSmartPointer<vtkRenderWindow>::New();
    vtkSmartPointer<vtkRenderWindowInteractor> iren = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    vtkSmartPointer<vtkAssembly> assembly = vtkSmartPointer<vtkAssembly>::New();

    for (auto &iter : all_cams) {
        vtkSmartPointer<vtkCubeSource> cube = vtkSmartPointer<vtkCubeSource>::New();
        cube->SetCenter(iter.t.at<dtype>(0, 0), iter.t.at<dtype>(0, 1), iter.t.at<dtype>(0, 2));
        cube->SetXLength(0.5);
        cube->SetYLength(0.5);
        cube->SetZLength(0.5);
        cube->Update();

        vtkSmartPointer<vtkPolyDataMapper> cube_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        cube_mapper->SetInputData(cube->GetOutput());
        vtkSmartPointer<vtkActor> cube_actor = vtkSmartPointer<vtkActor>::New();
        cube_actor->SetMapper(cube_mapper);
        cube_actor->GetProperty()->SetColor((colors->GetColor3d("Tomato").GetData()));
        assembly->AddPart(cube_actor);
    }
    vector<Point3> bound_coord = grid.Get_bound_coord();
    assembly->AddPart(Construct_lineActor(bound_coord[0], bound_coord[1]));
    assembly->AddPart(Construct_lineActor(bound_coord[1], bound_coord[2]));
    assembly->AddPart(Construct_lineActor(bound_coord[2], bound_coord[3]));
    assembly->AddPart(Construct_lineActor(bound_coord[3], bound_coord[0]));
    assembly->AddPart(Construct_lineActor(bound_coord[0], bound_coord[4]));
    assembly->AddPart(Construct_lineActor(bound_coord[4], bound_coord[5]));
    assembly->AddPart(Construct_lineActor(bound_coord[5], bound_coord[6]));
    assembly->AddPart(Construct_lineActor(bound_coord[6], bound_coord[7]));
    assembly->AddPart(Construct_lineActor(bound_coord[1], bound_coord[5]));
    assembly->AddPart(Construct_lineActor(bound_coord[2], bound_coord[6]));
    assembly->AddPart(Construct_lineActor(bound_coord[3], bound_coord[7]));
    assembly->AddPart(Construct_lineActor(bound_coord[4], bound_coord[7]));

    ren->AddActor(assembly);
    ren->SetBackground(colors->GetColor3d("Silver").GetData());
    renw->AddRenderer(ren);
    iren->SetRenderWindow(renw);
    renw->Render();
    iren->Start();

}

