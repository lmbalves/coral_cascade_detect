#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;
void detectAndDisplay( Mat frame );
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
int main( int argc, const char** argv )
{

    String face_cascade_name = "haarcascade_frontalface.xml";
    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) )
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    }
    int camera_device = 0;
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
        #pragma omp single
        {
        detectAndDisplay( frame );
        }

        if( waitKey(10) == 27 )
        {
            break; // escape
        }
    
    }
    return 0;
}
void detectAndDisplay( Mat frame )
{
    Mat frame_gray;

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    //-- Detect faces
    std::vector<Rect> faces;
    
    auto t1 = std::chrono::high_resolution_clock::now(); 
    face_cascade.detectMultiScale( frame_gray, faces );

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << duration << "\n";
    //-- Show what you got
    imshow( "Capture - Face detection", frame );
}
