#include <iostream>
#include <opencv2/core/core.hpp>

#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);

//#include <vtk-8.2/vtkSmartPointer.h>
//#include <vtk-8.2/vtkRenderWindow.h>
//#include <vtk-8.2/vtkRenderer.h>
//#include <vtk-8.2/vtkRenderWindowInteractor.h>
//#include <vtk-8.2/vtkCylinderSource.h>
//#include <vtk-8.2/vtkCylinder.h>
//#include <vtk-8.2/vtkNamedColors.h>
//#include <vtk-8.2/vtkPolyData.h>
//#include <vtk-8.2/vtkPolyDataMapper.h>
//#include <vtk-8.2/vtkActor.h>
//#include <vtk-8.2/vtkProperty.h>
#include <vtkActor.h>
#include <vtkCylinderSource.h>
#include <vtkNamedColors.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>

int main() {
    vtkSmartPointer<vtkNamedColors> colors =
            vtkSmartPointer<vtkNamedColors>::New();

    // Create a sphere
    vtkSmartPointer<vtkCylinderSource> cylinderSource =
            vtkSmartPointer<vtkCylinderSource>::New();
    cylinderSource->SetCenter(0.0, 0.0, 0.0);
    cylinderSource->SetRadius(5.0);
    cylinderSource->SetHeight(7.0);
    cylinderSource->SetResolution(100);

    // Create a mapper and actor
    vtkSmartPointer<vtkPolyDataMapper> mapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(cylinderSource->GetOutputPort());
    vtkSmartPointer<vtkActor> actor =
            vtkSmartPointer<vtkActor>::New();
    actor->GetProperty()->SetColor(colors->GetColor3d("Cornsilk").GetData());
    actor->SetMapper(mapper);

    //Create a renderer, render window, and interactor
    vtkSmartPointer<vtkRenderer> renderer =
            vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow =
            vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetWindowName("Cylinder");
    renderWindow->AddRenderer(renderer);
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
            vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    // Add the actor to the scene
    renderer->AddActor(actor);
    renderer->SetBackground(colors->GetColor3d("DarkGreen").GetData());

    // Render and interact
    renderWindow->Render();
    renderWindowInteractor->Start();

    return EXIT_SUCCESS;
}