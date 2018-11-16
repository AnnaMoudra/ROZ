#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
//#include <opencv2/core/mat.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;


class Image{

public:

    string filename;
    Mat img;
    vector<KeyPoint> keypoints;


    Image(string fn){
        this->filename = fn.c_str();
    }

    void loadImage(){
        this->img = imread(this->filename, IMREAD_COLOR);
        // Read in the image file
        int rows = this->img.rows;
        int cols = this->img.cols;
        int channels = this->img.channels();

        cout << "Image Rows = " << rows << endl;
        cout << "Image Columns = " << cols << endl;
        cout << "Image Channels = " << channels << endl;
    }

    void showImg(){
        namedWindow("Image Window", cv::WINDOW_AUTOSIZE);
        // Create a window of the same size as the image for display
        imshow("Image Window", this->img);
        // Show our image inside the window
        waitKey(0);
        // Wait for a keystroke in the window
    }

    void imgToLAB(){

        cvtColor(this->img, this->img, CV_BGR2Lab);
    }

    void preProcessing(){
        //TODO


        //fix edges??

    }

    void EF(){
        int minHessian = 400;
        Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );

    }

};

#endif
