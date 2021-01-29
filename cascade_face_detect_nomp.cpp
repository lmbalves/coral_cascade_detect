#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace cv;

void detectAndDisplay(Mat frame);

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

int main(int argc, const char **argv)
{

    String face_cascade_name = "haarcascade_frontalface_alt.xml";
    String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
    //-- 1. Load the cascades
    if (!face_cascade.load(face_cascade_name))
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };
    if (!eyes_cascade.load(eyes_cascade_name))
    {
        cout << "--(!)Error loading eyes cascade\n";
        return -1;
    };
    int camera_device = 0;
    VideoCapture capture;
    //-- 2. Read the video stream
    capture.open(camera_device);
    if (!capture.isOpened())
    {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }
    Mat frame;
    while (capture.read(frame))
    {
        if (frame.empty())
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }
        //-- 3. Apply the classifier to the frame

        detectAndDisplay(frame);

        if (waitKey(10) == 27)
        {
            break; // escape
        }
    }
    return 0;
}
void detectAndDisplay(Mat frame)
{
    Mat frame_gray;

    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    //-- Detect faces

    std::vector<Rect> faces;
    std::vector<Rect> eyes;

    auto t1 = chrono::high_resolution_clock::now();
    
    face_cascade.detectMultiScale(frame_gray, faces);

    

    auto t2 = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
    std::cout << duration << "\n";

    for (size_t j = 0; j < faces.size(); j++)
    {
        eyes_cascade.detectMultiScale(frame_gray, eyes);
        Point eye_center(eyes[j].x + eyes[j].width / 2, eyes[j].y + eyes[j].height / 2);
        int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
        circle(frame, eye_center, radius, Scalar(255, 0, 0), 4);
    }
    for (size_t i = 0; i < faces.size(); i++)
    {
        Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
        ellipse(frame, center, Size(faces[i].width / 1.5, faces[i].height), 0, 0, 360, Scalar(255, 0, 0), 4);
    }
    imshow("Capture - Face detection", frame);
}