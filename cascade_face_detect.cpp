#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <chrono>
#include <omp.h>
using namespace std;
using namespace cv;
void detectAndDisplay( Mat frame, int NUM_THREADS );

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv,
                             "{help h||}"
                             "{threads|1|number of threads.}");
    parser.about( "\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face) in a video stream.\n\n");
    parser.printMessage();
    String face_cascade_name = "haarcascade_frontalface_alt_tree.xml"/* samples::findFile( parser.get<String>("face_cascade") )*/;
    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) )
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };
    int camera_device = 0;
    int NUM_THREADS = parser.get<int>("threads");
    VideoCapture capture;
    //-- 2. Read the video stream
    capture.open( camera_device );
    if ( ! capture.isOpened() )
    {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }
    Mat frame;
    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }
        //-- 3. Apply the classifier to the frame

        detectAndDisplay( frame, NUM_THREADS );

        if( waitKey(10) == 27 )
        {
            break; // escape
        }
    }
    return 0;
}
void detectAndDisplay( Mat frame, int NUM_THREADS )
{
    Mat frame_gray;

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    std::vector<Rect> faces;
    std::vector<Rect> eyes;
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel num_threads( NUM_THREADS)
    {
                face_cascade.detectMultiScale( frame_gray, faces);
                        printf("Hello World... from thread = %d\n", 
               omp_get_thread_num()); 
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << duration << "\n";
    for (size_t i = 0; i < faces.size(); i++)
    {
        Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
        ellipse(frame, center, Size(faces[i].width / 1.2, faces[i].height/1.2), 0, 0, 360, Scalar(255, 0, 0), 4);
    }
    imshow("Capture - Face detection", frame);
}