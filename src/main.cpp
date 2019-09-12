#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

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

using namespace cv;
int main() {
    Mat img = imread("../res/images/image_000.jpg", 0);
    namedWindow("test", 0);
    imshow("test", img);
    waitKey(0);
    return 0;
}