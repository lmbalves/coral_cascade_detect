#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>

//(1) include face header
#include "opencv2/face.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

//(2) include face header
#include "opencv2/objdetect.hpp"
#include <iostream>

using namespace cv;
using namespace std;
using namespace ml;
using namespace cv::face;

//(3) Global variables 
Ptr<Facemark> facemark;

CascadeClassifier faceDetector;

void process(Mat img, Mat imgcol) {

    vector<Rect> faces;

    faceDetector.detectMultiScale(img, faces);

    vector< vector<Point2f> > shapes;

    Mat imFace;

    if (faces.size() != 0) {

        for (size_t i = 0; i < faces.size(); i++)
        {
            cv::rectangle(imgcol, faces[i], Scalar(255, 0, 0));
            imFace = imgcol(faces[i]);
            resize(imFace, imFace, Size(imFace.cols * 5, imFace.rows * 5));
            faces[i] = Rect(faces[i].x = 0, faces[i].y = 0, faces[i].width * 5,
            (faces[i].height) * 5);
        }
        vector < Rect > measures;

        if (facemark->fit(imFace, faces, shapes))
        {
            for (unsigned long i = 0; i < faces.size(); i++)
            {
                for (unsigned long k = 0; k < shapes[i].size(); k++)
                {
                    cv::circle(imFace, shapes[i][k], 5, cv::Scalar(0, 0, 255),FILLED);
                }
            }
        }

        namedWindow("Detected_shape");
        imshow("Detected_shape", imFace);
        waitKey(5);
    }
    else {
        cout << "Faces not detected." << endl;
    }
}

int main()
{
    facemark = FacemarkLBF::create();
    facemark->loadModel("lbfmodel.yaml.txt");
    faceDetector.load("haarcascade_frontalface_alt2.xml");
    cout << "Loaded model" << endl;

    VideoCapture cap(0);
    int initialized = 0;

    for (;;)
    {
        if (!cap.isOpened())
        {
            cout << "Video Capture Fail" << endl;

            break;
        }
        else {
            Mat img;
            Mat imgbw;
            cap >> img;  
            resize(img, img, Size(460, 460), 0, 0, INTER_LINEAR_EXACT);
            cvtColor(img, imgbw, COLOR_BGR2GRAY);           
            process(imgbw, img);


            namedWindow("Live", WINDOW_AUTOSIZE);
            setMouseCallback("Live", CallBackF, 0);
            imshow("Live", img);
            waitKey(5);
        }
    }
}