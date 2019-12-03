//
// Created by himalaya on 10/3/19.
//

#include "display.h"
#include "evolution.h"
#include <omp.h>

#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);
VTK_MODULE_INIT(vtkRenderingFreeType);

#include <vtkSTLWriter.h>
#include <vtkSTLReader.h>
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
#include <vtkCaptionActor2D.h>
#include <vtkTextProperty.h>
#include <vtkAxesActor.h>
#include <vtkCommand.h>
#include <vtkCallbackCommand.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkMarchingCubes.h>
#include <vtkImageImport.h>
#include <iostream>

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

//void CreateData(vtkImageData* data) {
//    data->SetExtent(-25, 25, -25, 25, -25, 25);
//    data->AllocateScalars(VTK_DOUBLE, 1);
//    int *extent = data->GetExtent();
//    for (int z = extent[4]; z<= extent[5]; z++){
//        for (int y = extent[2]; y <= extent[3]; y++) {
//            for (int x = extent[0]; x <= extent[1]; x++) {
//                double *pixel = static_cast<double *>(data->GetScalarPointer(x, y, z));
//                pixel[0] = sqrt(pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0)) - 10;
//            }
//        }
//    }
//}

/**
 * transfrom c-order array into vtk-order
 */
void Transform_phi(const BoundingBox &box, dtype *new_phi)
{
    auto depth = box.grid3d->_depth;
    auto width = box.grid3d->_width;
    auto height = box.grid3d->_height;
    //// change the storing order for vtk
#pragma omp parallel for default(none) shared(depth, width, height, new_phi, box)
    for (IdxType z = 0; z < depth; z++) {
        for (IdxType y = 0; y < width; y++) {
            for (IdxType x = 0; x < height; x++) {
                new_phi[x + y * height + z * height * width] = box.grid3d->phi[box.grid3d->Index(x, y, z)];
            }
        }
    }
}

vtkSmartPointer<vtkActor> Render_surface(const BoundingBox &box, double level_set_val)
{
    auto total_num_points = box.grid3d->_height * box.grid3d->_width * box.grid3d->_depth;
    assert (total_num_points > 0);
    dtype *new_phi = new dtype [total_num_points];
    auto depth = box.grid3d->_depth;
    auto width = box.grid3d->_width;
    auto height = box.grid3d->_height;
    //// change the storing order for vtk
#pragma omp parallel for default(none) shared(depth, width, height, new_phi, box)
    for (IdxType z = 0; z < depth; z++) {
        for (IdxType y = 0; y < width; y++) {
            for (IdxType x = 0; x < height; x++) {
                new_phi[x + y * height + z * height * width] = box.grid3d->phi[box.grid3d->Index(x, y, z)];
            }
        }
    }
    vtkSmartPointer<vtkFloatArray> phi_arr = vtkSmartPointer<vtkFloatArray>::New();
    phi_arr->SetArray(new_phi,total_num_points, 1);

    auto phi_data = vtkSmartPointer<vtkImageData>::New();
//    auto phi_data = vtkSmartPointer<vtkImageImport>::New();
    phi_data->GetPointData()->SetScalars(phi_arr);
    phi_data->SetDimensions(height, width, depth);
    auto bound = box.Get_bound_coord();
    phi_data->SetOrigin(bound[0].x, bound[0].y, bound[0].z);
    phi_data->SetSpacing(box.resolution, box.resolution, box.resolution);

//    for (int z = 0; z < box.grid3d->height; z++) {
//        for (int y = 0; y < box.grid3d->width; y++) {
//            for (int x = 0; x < box.grid3d->length; x++) {
//                float *pixel = static_cast<float *>(phi_data->GetScalarPointer(x, y, z));
//                cout << pixel[0] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }
//    auto phi_data = vtkSmartPointer<vtkImageData>::New();
//    CreateData(phi_data);
//    phi_data->SetSpacing(1, 1, 1);

//    phi_data->CopyImportVoidPointer(box.grid3d->phi.data(), total_num_points * sizeof(dtype) / sizeof(unsigned char));
//    phi_data->SetWholeExtent(0, box.grid3d->height - 1, 0, box.grid3d->width - 1, 0, box.grid3d->length - 1);
//    cout << phi_data->GetWholeExtent()[1] << endl;
//    phi_data->SetDataSpacing(box.resolution, box.resolution, box.resolution);
//    phi_data->SetDataOrigin(bound[0].x, bound[0].y, bound[0].z);
//    phi_data->SetDataExtentToWholeExtent();
//    phi_data->SetDataScalarTypeToFloat();
//    phi_data->SetNumberOfScalarComponents(1);
//    cout << phi_data->GetNumberOfScalarComponents() << endl;
    //    phi_data->Update();
//    for (int z = 0; z < box.grid3d->height; z++) {
//        for (int y = 0; y < box.grid3d->width; y++) {
//            for (int x = 0; x < box.grid3d->length; x++) {
//                float *pixel = static_cast<float *>(phi_data->GetImportVoidPointer());
//                cout << pixel[x + y * box.grid3d->length + z * box.grid3d->length * box.grid3d->width] << " ";
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }


    auto isosurface = vtkSmartPointer<vtkMarchingCubes>::New();
    isosurface->SetInputData(phi_data);
    isosurface->ComputeGradientsOn();
    isosurface->ComputeNormalsOn();
    isosurface->ComputeScalarsOff();
    isosurface->SetValue(0, level_set_val);

    auto surface_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    surface_mapper->SetInputConnection(isosurface->GetOutputPort());
    surface_mapper->ScalarVisibilityOn();

    auto surface_actor = vtkSmartPointer<vtkActor>::New();
    surface_actor->SetMapper(surface_mapper);

    return surface_actor;
}

class vtkTimerCallback : public vtkCommand
{
public:
    BoundingBox *p_box;
    vector<Camera> *p_all_cams;
    dtype *data;
    vtkMarchingCubes *iso;
    vtkPoints *points;
    vtkCellArray *vertices;
    vtkPolyDataMapper *pt_mapper;
    vtkTimerCallback():p_box(nullptr), p_all_cams(nullptr) {}
    static vtkTimerCallback* New()
    {
        vtkTimerCallback *cb = new vtkTimerCallback;
        return cb;
    }
    virtual void Execute(vtkObject* caller, unsigned long eventId, void* vtkNotUsed(callData))
    {
        auto *iren = dynamic_cast<vtkRenderWindowInteractor *>(caller);
        if (this->trigger_count != 0) {
            assert(p_box != nullptr && p_all_cams != nullptr);
            cout << this->trigger_count++ << " callback" << endl;
            Evolve(*p_box, *p_all_cams);
            Transform_phi(*p_box, data);
            iso->Modified();

//            pt_mapper->Modified();
            iren->GetRenderWindow()->Render();
        }
        else {
            iren->GetRenderWindow()->Render();
            this->trigger_count++;
        }

    }
    void Set_data(BoundingBox &box, vector<Camera> &all_cams)
    {
        this->p_box = &box;
        this->p_all_cams = &all_cams;
    }

private:
    int trigger_count = 0;
};

void KeypressCallbackFunction ( vtkObject* caller, long unsigned int vtkNotUsed(eventId), void* vtkNotUsed(clientData), void* vtkNotUsed(callData) )
{
//    std::cout << "Keypress callback" << std::endl;

    auto *iren = static_cast<vtkRenderWindowInteractor*>(caller);

//    std::cout << "Pressed: " << iren->GetKeySym() << std::endl;
    char *temp_key = iren->GetKeySym();
    string key(temp_key);
    if (key == "Return") {
        iren->CreateRepeatingTimer(10);
    }
    else if (key == "p") {
        iren->DestroyTimer();
    }
}

#if USE_NEW
void Show_3D(vector<Camera> &all_cams, BoundingBox &box)
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
    vector<Point3> bound_coord = box.Get_bound_coord();
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

    // add axes
//    vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();
//    axes->SetTotalLength(10, 10, 10);
//    axes->GetXAxisCaptionActor2D()->GetCaptionTextProperty()->SetFontSize(1);
//    axes->GetYAxisCaptionActor2D()->GetCaptionTextProperty()->SetFontSize(1);
//    axes->GetZAxisCaptionActor2D()->GetCaptionTextProperty()->SetFontSize(1);
//    ren->AddActor(axes);

    // use marching cube to render surface
    double level_val = 0;
    auto total_num_points = box.grid3d->_height * box.grid3d->_width * box.grid3d->_depth;
    assert (total_num_points > 0);
    dtype *new_phi = new dtype [total_num_points];
    auto depth = box.grid3d->_depth;
    auto width = box.grid3d->_width;
    auto height = box.grid3d->_height;
    Transform_phi(box, new_phi);

    vtkSmartPointer<vtkFloatArray> phi_arr = vtkSmartPointer<vtkFloatArray>::New();
    phi_arr->SetArray(new_phi,total_num_points, 0);

    auto phi_data = vtkSmartPointer<vtkImageData>::New();
//    auto phi_data = vtkSmartPointer<vtkImageImport>::New();
    phi_data->GetPointData()->SetScalars(phi_arr);
    phi_data->SetDimensions(height, width, depth);
    auto bound = box.Get_bound_coord();
    phi_data->SetOrigin(bound[0].x, bound[0].y, bound[0].z);
    phi_data->SetSpacing(box.resolution, box.resolution, box.resolution);

    auto isosurface = vtkSmartPointer<vtkMarchingCubes>::New();
    isosurface->SetInputData(phi_data);
    isosurface->ComputeGradientsOff();
    isosurface->ComputeNormalsOn();
    isosurface->ComputeScalarsOff();
    isosurface->SetValue(0, level_val);

    auto surface_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    surface_mapper->SetInputConnection(isosurface->GetOutputPort());
    surface_mapper->ScalarVisibilityOn();

    vtkSmartPointer<vtkSTLWriter> stlWriter = vtkSmartPointer<vtkSTLWriter>::New();
    stlWriter->SetFileName("final_data.stl");
    stlWriter->SetInputConnection(isosurface->GetOutputPort());

    auto surface_actor = vtkSmartPointer<vtkActor>::New();
    surface_actor->SetMapper(surface_mapper);

    // render nonvisible point
//    auto points = vtkSmartPointer<vtkPoints>::New();
//    auto vertices = vtkSmartPointer<vtkCellArray>::New();
//    auto pt_polydata = vtkSmartPointer<vtkPolyData>::New();
//    pt_polydata->SetPoints(points);
//    pt_polydata->SetVerts(vertices);
//    auto pt_mapper = vtkSmartPointer<vtkPolyDataMappe  r>::New();
//    pt_mapper->SetInputData(pt_polydata);
//    auto pt_actor = vtkSmartPointer<vtkActor>::New();
//    const auto &grid3d = box.grid3d;
//    for (IdxType i = 0; i < box.grid3d->_height; i++) {
//        for (IdxType j = 0; j < box.grid3d->_width; j++) {
//            for (IdxType k = 0; k < box.grid3d->_depth; k++) {
//                if (box.visibility_arr[0].psi[box.visibility_arr[0].Index(i, j, k)] < 0) {
//                    const auto &p = grid3d->coord[grid3d->Index(i, j, k)];
//                    auto id = points->InsertNextPoint(p.val);
//                    vertices->InsertNextCell(1);
//                    vertices->InsertCellPoint(id);
//                }
//            }
//        }
//    }
//    pt_actor->SetMapper(pt_mapper);
//    pt_actor->GetProperty()->SetColor(255, 0, 0);
//
//    ren->AddActor(pt_actor);

//    ren->AddActor(Render_surface(box, 0));
    ren->AddActor(surface_actor);
    ren->AddActor(assembly);
    ren->SetBackground(colors->GetColor3d("Silver").GetData());
    renw->AddRenderer(ren);
    renw->SetSize(800, 800);
    iren->SetRenderWindow(renw);
    renw->Render();

    iren->Initialize();

    vtkSmartPointer<vtkCallbackCommand> keypressCallback =
            vtkSmartPointer<vtkCallbackCommand>::New();
    keypressCallback->SetCallback ( KeypressCallbackFunction );
    iren->AddObserver(vtkCommand::KeyPressEvent, keypressCallback, 0);


    auto cb = vtkSmartPointer<vtkTimerCallback>::New();
    cb->Set_data(box, all_cams);
    cb->data = new_phi;
    cb->iso = isosurface;
//    cb->points = points;
//    cb->vertices = vertices;
//    cb->pt_mapper = pt_mapper;
    iren->AddObserver(vtkCommand::TimerEvent, cb, 1.);
//    iren->CreateRepeatingTimer(1000);
    iren->Start();
    stlWriter->Write();
}
#else
void Show_3D(const vector<Camera> &all_cams, const BoundingBox &box)
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
    vector<Point3> bound_coord = box.Get_bound_coord();
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

    // add axes
//    vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();
//    axes->SetTotalLength(10, 10, 10);
//    axes->GetXAxisCaptionActor2D()->GetCaptionTextProperty()->SetFontSize(1);
//    axes->GetYAxisCaptionActor2D()->GetCaptionTextProperty()->SetFontSize(1);
//    axes->GetZAxisCaptionActor2D()->GetCaptionTextProperty()->SetFontSize(1);
//    ren->AddActor(axes);

    // render nonvisible point
    auto points = vtkSmartPointer<vtkPoints>::New();
    auto vertices = vtkSmartPointer<vtkCellArray>::New();
    auto pt_polydata = vtkSmartPointer<vtkPolyData>::New();
    pt_polydata->SetPoints(points);
    pt_polydata->SetVerts(vertices);
    auto pt_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    pt_mapper->SetInputData(pt_polydata);
    auto pt_actor = vtkSmartPointer<vtkActor>::New();
    const auto &grid3d = box.grid3d;
    for (IdxType i = 0; i < box.grid3d->_height; i++) {
        for (IdxType j = 0; j < box.grid3d->_width; j++) {
            for (IdxType k = 0; k < box.grid3d->_depth; k++) {
                if (box.visibility_arr[0].psi[box.visibility_arr[0].Index(i, j, k)] < 0) {
                    const auto &p = grid3d->coord[grid3d->Index(i, j, k)];
                    auto id = points->InsertNextPoint(p.val);
                    vertices->InsertNextCell(1);
                    vertices->InsertCellPoint(id);
                }
            }
        }
    }
    pt_actor->SetMapper(pt_mapper);
    pt_actor->GetProperty()->SetColor(255, 0, 0);

//    auto line = vtkSmartPointer<vtkLineSource>::New();
//    Vec3 cam(all_cams[0].t);
//    line->SetPoint1(cam.val);
//    line->SetPoint2(grid3d->coord[grid3d->Index(grid3d->_height / 4 * 3 - 2, grid3d->_width - 37, grid3d->_depth / 2 )].val);
//    auto line_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
//    line_mapper->SetInputConnection(line->GetOutputPort());
//    auto line_actor = vtkSmartPointer<vtkActor>::New();
//    line_actor->SetMapper(line_mapper);
//    line_actor->GetProperty()->SetColor(0, 1, 0);
//    ren->AddActor(line_actor);

    ren->AddActor(pt_actor);

    ren->AddActor(Render_surface(box, 0));
    ren->AddActor(assembly);
    ren->SetBackground(colors->GetColor3d("Silver").GetData());
    renw->AddRenderer(ren);
    renw->SetSize(800, 800);
    iren->SetRenderWindow(renw);
    renw->Render();


    iren->Start();
}
#endif

void Show_3D(std::string file_name)
{
    vtkSmartPointer<vtkSTLReader> reader =
            vtkSmartPointer<vtkSTLReader>::New();
    reader->SetFileName(file_name.c_str());
    reader->Update();

    vtkSmartPointer<vtkPolyDataMapper> mapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());

    vtkSmartPointer<vtkActor> actor =
            vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    vtkSmartPointer<vtkRenderer> renderer =
            vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow =
            vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
            vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    renderer->AddActor(actor);

    renderWindow->Render();
    renderWindowInteractor->Start();
}