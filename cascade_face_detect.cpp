
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <chrono>
#include <omp.h>
using namespace std;
using namespace cv;
void detectAndDisplay( Mat frame );

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
int main( int argc, const char** argv )
{


   /* CommandLineParser parser(argc, argv,
                             "{help h||}"
                             "{face_cascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
                             "{eyes_cascade|data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|Path to eyes cascade.}"
                             "{camera|0|Camera device number.}");
    parser.about( "\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
                  "You can use Haar or LBP features.\n\n" );
    parser.printMessage();*/
    String face_cascade_name = "haarcascade_frontalface_alt.xml"/* samples::findFile( parser.get<String>("face_cascade") )*/;
    String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml"/*samples::findFile( parser.get<String>("eyes_cascade") )*/;
    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) )
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };
    if( !eyes_cascade.load( eyes_cascade_name ) )
    {
        cout << "--(!)Error loading eyes cascade\n";
        return -1;
    };
    int camera_device = 0/*parser.get<int>("camera")*/;
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

        detectAndDisplay( frame );

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
    cout << "Total Devices: " << omp_get_num_devices() << endl;
    //omp_set_default_device
    omp_set_num_threads(2);
    std::vector<Rect> faces;
    std::vector<Rect> eyes;
    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                face_cascade.detectMultiScale( frame_gray, faces);
            }
            #pragma omp section
            {                    
                eyes_cascade.detectMultiScale( frame_gray, eyes );
            }               
        }

        #pragma omp critical
        {
            auto t2 = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
            std::cout << duration << "\n";

            for ( size_t j = 0; j < eyes.size(); j++ )
            {
                Point eye_center( eyes[j].x + eyes[j].width/2, eyes[j].y + eyes[j].height/2 );
                int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
                circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4 );
            }
            for ( size_t i = 0; i < faces.size(); i++ )
            {
                Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
                ellipse( frame, center, Size( faces[i].width/1.5, faces[i].height ), 0, 0, 360, Scalar( 255, 0, 0 ), 4 );
            }
            imshow( "Capture - Face detection", frame );
        }
    }
}