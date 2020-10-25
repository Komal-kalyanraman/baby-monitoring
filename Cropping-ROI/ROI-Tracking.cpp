
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <opencv2\imgproc.hpp>
#include <opencv2\objdetect.hpp>

# define camera_no 0

# define TakeSnap 0
# define CropImage 1
# define CroppedVideoFeed 2

using namespace std;

// Globals
bool finished = false;
cv::Mat img, ROI, mask;
vector<cv::Point> vertices;
int mode = 0;

cv::Point start_length = cv::Point(10, 20);

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_RBUTTONDOWN) {
        cout << "Right mouse button clicked at (" << x << ", " << y << ")" << endl;
        if (vertices.size() < 2) {
            cout << "You need a minimum of three points!" << endl;
            return;
        }
        // Close polygon
        line(img, vertices[vertices.size() - 1], vertices[0], cv::Scalar(255, 255, 0), 3);

        // Mask is black with white where our ROI is
        mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
        vector<vector<cv::Point>> pts{ vertices };
        fillPoly(mask, pts, cv::Scalar(255, 255, 255));
        img.copyTo(ROI, mask);
        finished = true;

        return;
    }
    if (event == cv::EVENT_LBUTTONDOWN) {
        cout << "Left mouse button clicked at (" << x << ", " << y << ")" << endl;
        if (vertices.size() == 0) {
            // First click - just draw point
            img.at<cv::Vec3b>(x, y) = cv::Vec3b(255, 0, 0);
        }
        else {
            // Second, or later click, draw line to previous vertex
            line(img, cv::Point(x, y), vertices[vertices.size() - 1], cv::Scalar(255, 255, 0), 3);
        }
        vertices.push_back(cv::Point(x, y));
        return;
    }
}

int main(int argc, char** argv)
{
    cv::Mat src, crop;
    cv::VideoCapture camera(camera_no);
    char key = 'r';
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    cv::CascadeClassifier detector;
    //string cascadeName = "C:/Users/koy5cob/Desktop/MoonShot_1/ComputerVision/BabyMonitoring/IntrusionDetection/Harrcascade_files/haarcascade_fullbody.xml";
    string cascadeName = "C:/Users/koy5cob/Desktop/MoonShot_1/ComputerVision/BabyMonitoring/IntrusionDetection/Harrcascade_files/haarcascade_frontalface_alt2.xml";
    bool loaded = detector.load(cascadeName);

    // Tunable Parameters of detectMultiscale Cascade Classifier
    int groundThreshold = 2;
    double scaleStep = 1.1;
    cv::Size minimalObjectSize(20, 20);
    cv::Size maximalObjectSize(200, 200);

    // Vector of returned faces
    vector< cv::Rect_<int> > found;
    //int track = 0;
    cv::Point center, track;
    vector<cv::Point> pointList;

    cv::Mat image_grey;
    /////////////////////////////////////////////////////////////////////////////////////////////

    camera.set(3, 640);	//for setting o/p image resolution
    camera.set(4, 480);

    string str = "click c to take snap";

    while(true)
    {
        while (mode == TakeSnap)
        {
            camera >> src;

            //if fail to read the image
            if (src.empty())
            {
                cout << "Error loading the image" << endl;
                return -1;
            }

            cv::putText(src, str, start_length,
                cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

            //Create a window
            cv::namedWindow("My Window", 1);

            //show the image
            cv::imshow("My Window", src);

            key = cv::waitKey(1);

            if (key == 'c' || key == 'C')
            {
                cv::imwrite("test.jpg", src);

                cv::imshow("Crop Image", src);
                mode= CropImage;
                cv::destroyAllWindows();
            }

            if (key == 'q' || key == 'Q')
                break;
        }
        
        while (mode == CropImage)
        {
            img = cv::imread("C:/Users/koy5cob/Desktop/MoonShot_1/ComputerVision/BabyMonitoring/IntrusionDetection/IntrusionDetection/test.jpg");

            // Check it loaded
            if (img.empty())
            {
                cout << "Error loading the image" << endl;
                exit(1);
            }

            //Create a window
            cv::namedWindow("ImageDisplay", 1);

            // Register a mouse callback
            cv::setMouseCallback("ImageDisplay", CallBackFunc, nullptr);

            // Main loop
            while (!finished) {
                cv::imshow("ImageDisplay", img);
                cv::waitKey(1);
            }

            // Show results
            cv::imshow("Result", ROI);
            cv::waitKey(300);

            //cv::destroyWindow("ImageDisplay");
            cv::destroyAllWindows();

            mode = CroppedVideoFeed;

            /*key = cv::waitKey(10);

            if (key == 'q' || key == 'Q')
                break;*/
        }

        while (mode == CroppedVideoFeed)
        {
            camera >> src;

            //if fail to read the image
            if (src.empty())
            {
                cout << "Error loading the image" << endl;
                return -1;
            }

            //Create a window
            src.copyTo(crop, mask);

            //show the image
            //cv::imshow("Full feed", src);
            //cv::imshow("Cropped feed", crop);

            ////////////////////////////////////////////////////////////////
            cv::cvtColor(crop, image_grey, cv::COLOR_BGR2GRAY);
            found.clear();

            // Detect faces
            detector.detectMultiScale(image_grey, found, scaleStep, groundThreshold, 0 | cv::CASCADE_SCALE_IMAGE, minimalObjectSize, maximalObjectSize);

            if (found.size() > 0) {
                for (int i = 0; i <= found.size() - 1; i++) {

                    cv::rectangle(crop, found[i].br(), found[i].tl(), cv::Scalar(0, 255, 0), 5, 8, 0);
                    center = cv::Point((found[i].x + found[i].width / 2), (found[i].y + found[i].height / 2));
                    cv::circle(crop, center, 5, cv::Scalar(0, 0, 255), -1);
                }
            }

            cv::imshow("Face detection", crop);
            ////////////////////////////////////////////////////////////////

            key = cv::waitKey(1);

            if (key == 'q' || key == 'Q')
                break;
        }
        break;
    }
    return 0;
}
