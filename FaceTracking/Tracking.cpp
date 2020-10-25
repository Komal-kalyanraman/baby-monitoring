
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <string>

using namespace std;

# define camera_no 0    // Camera which is being used

#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480

cv::VideoCapture cap;

int main(int argc, char** argv)
{
    int count = 0;

    cap.open(camera_no);
    cap.set(3, IMAGE_WIDTH);
    cap.set(4, IMAGE_HEIGHT);

    cv::Mat img;

    // Load cascate classifier placed in folder
    cv::CascadeClassifier detector;
    
    //string cascadeName = "C:/Users/koy5cob/Desktop/MoonShot_1/ComputerVision/EmotionDetection/FaceDetection/Harrcascade_files/haarcascade_frontalface_alt.xml";
    //string cascadeName = "C:/Users/koy5cob/Desktop/MoonShot_1/ComputerVision/EmotionDetection/FaceDetection/Harrcascade_files/haarcascade_frontalface_alt_tree.xml";
    string cascadeName = "C:/Users/koy5cob/Desktop/MoonShot_1/ComputerVision/EmotionDetection/FaceDetection/Harrcascade_files/haarcascade_frontalface_alt2.xml";
    //string cascadeName = "C:/Users/koy5cob/Desktop/MoonShot_1/ComputerVision/EmotionDetection/FaceDetection/Harrcascade_files/haarcascade_frontalface_default.xml";
    
    bool loaded = detector.load(cascadeName);
    // Parameters of detectMultiscale Cascade Classifier
    int groundThreshold = 2;
    double scaleStep = 1.1;
    cv::Size minimalObjectSize(80, 80);
    cv::Size maximalObjectSize(200, 200);

    // Vector of returned faces
    vector< cv::Rect_<int> > found;
    //int track = 0;
    cv::Point center, track;
    vector<cv::Point> pointList;
    cv::Point start_length = cv::Point(10, 20);
    cv::Point start_alert = cv::Point(10, 40);

    for (;;)
    {

        cap >> img;

        // Convert input to greyscale
        cv::Mat image_grey, thr;
        cv::cvtColor(img, image_grey, cv::COLOR_BGR2GRAY);
        cv::threshold(image_grey, thr, 100, 255, cv::THRESH_BINARY);

        found.clear();

        // Detect faces
        detector.detectMultiScale(image_grey, found, scaleStep, groundThreshold, 0 | cv::CASCADE_SCALE_IMAGE, minimalObjectSize, maximalObjectSize);

        // Draw the results into mat retrieved from webcam
        if (found.size() > 0) {
            for (int i = 0; i <= found.size() - 1; i++) {

                cv::rectangle(img, found[i].br(), found[i].tl(), cv::Scalar(0, 255, 0), 5, 8, 0);
                center = cv::Point((found[i].x + found[i].width/2), (found[i].y + found[i].height/2));
                cv::circle(img, center, 5, cv::Scalar(0, 0, 255), -1);

                if (pointList.size() < 20) {
                    pointList.push_back(center);

                }

                else {
                    
                    pointList.erase(pointList.end() - 20);
                    //erase(vec.end() - 1);
                    //pointList.pop_back();
                    //pointList.
                    pointList.push_back(center);
                }

                if (pointList.size() == 20) {
                    for (int n = 0; n < pointList.size(); n++)
                    {
                        cv::circle(img, pointList[n], 5, cv::Scalar(0, 0, 255), -1);
                    }
                    cv::Point diff = pointList[19] - pointList[0];
                    int distance = cv::sqrt(diff.x * diff.x + diff.y * diff.y);
                    string length = to_string(distance);
                    string str = "Distance moved = " + length;
                    cv::putText(img, str, start_length,
                        cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

                    if (distance > 40) {
                        string alert = "Too much movement";
                        cv::putText(img, alert, start_alert,
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                    }
                    else {
                        string alert = "No movement";
                        cv::putText(img, alert, start_alert,
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                    }

                }
            }
        }

        cv::imshow("Face detection", img);
        cv::waitKey(1);
    }
    return 0;
}
