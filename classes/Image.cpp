#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;
# define M_PI           3.14159265358979323846  /* pi */

class Image{

public:

    string filename;
    Mat original;
    Mat img;
    vector<KeyPoint> keypoints;
    Mat descriptors;
    int classNo = -1;
    MatND histogram;
    //gabor features
    vector<Mat> gabor;
    //gabor feature vector
    vector<int> gaborFeatures;


    Image(string fn){
        this->filename = fn.c_str();
        getClassNumber();
    }

    void loadImage(){
        this->img = imread(this->filename, IMREAD_COLOR);
        this->img.copyTo(this->original);
        // Read in the image file
        int rows = this->img.rows;
        int cols = this->img.cols;
        int channels = this->img.channels();
    }

    void cropImg(){
        Rect myROI(240, 0, 120, 70); //56.4
        //this->original = this->img;
        this->img = this->img(myROI);
    }

    void drawHistChange(){
        Rect myROI(240, 0, 120, 70); //56.4
        Mat tmp = this->img;
        this->img = this->original;
        rectangle(this->img, myROI, cv::Scalar(0, 255, 0));
        this->showImg();
        this->img = tmp;
    }

    void setOriginal(){
        this->img = this->original;
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


    }

    //todo EXTRACT ALL
    void extractGaborFeatures(Image * img){
        //8 rotations 1 setup = 8*1 response matrices
        vector<int> loc_energy;
        vector<int> mean_amplitude;
        for(auto g : img->gabor){
            //where g is one Gabor-filtered image (img->img)
            // g is a response matrice - My image convolved with Gabor feature (gabor filter)
            // feature vector: Local Energy,Mean Amplitude,Phase Amlitude or Orientation whose local has maximumEnergy

            //Local Energy = summing up the squared value of each matrix value from a response matrix
            //SUM(pixel^2)
            int sum = 0;
            for(int i=0; i<g.rows; i++) {
                for (int j = 0; j < g.cols; j++) {
                    int pixel = g.at<uchar>(i, j);
                    int pixel_p2 = pixel * pixel;
                    sum += pixel_p2;
                }
            }
            loc_energy.push_back(sum);
            //Mean Amplitude = sum of absolute values of each matrix value from a response matrix
            sum  = 0;
            for(int i=0; i<g.rows; i++) {
                for (int j = 0; j < g.cols; j++) {
                    int pixel = g.at<uchar>(i, j);
                    int pixel_abs = abs(pixel);
                    sum += pixel;
                }
            }
            mean_amplitude.push_back(sum);
        }
        vector<int> all;
        all.reserve( loc_energy.size() + mean_amplitude.size() ); // preallocate memory
        all.insert( all.end(), loc_energy.begin(), loc_energy.end() );
        all.insert( all.end(), mean_amplitude.begin(), mean_amplitude.end() );

        //return ft vector back ti Image
        img->gaborFeatures = all;
    }

    void extractHistogram(){
        Mat hsv_base;
        cvtColor(this->img, hsv_base, COLOR_BGR2HSV);

        int h_bins = 50; int s_bins = 60;
        int histSize[] = { h_bins, s_bins };
        float h_ranges[] = { 0, 180 };
        float s_ranges[] = { 0, 256 };
        const float* ranges[] = { h_ranges, s_ranges };
        int channels[] = { 0, 1 };


        calcHist( &hsv_base, 1, channels, Mat(), this->histogram, 2, histSize, ranges, true, false );
        normalize( this->histogram, this->histogram, 0, 1, NORM_MINMAX, -1, Mat() );

    }

    void EF_ORB(){};


    void EF_GABOR() {
        Mat gaborOut, outAsFloat, gaborImage;
        Mat tmp, outImg;
        this->img.copyTo(tmp);

        //get grayscale
        cv::cvtColor(tmp, outImg, CV_BGR2GRAY);
        //pixel in range 0.-1.
        outImg.convertTo(outAsFloat, CV_32F);
        //todo
        //8 rotaci, jeste SCALE!!!
        for (int j = 1; j < 21; j += 4) {
            for (int i = -180; i < 180; i += 45) {
                int size = 37;
                int sigma = 2;
                double theta = i;
                //kernel size, sigma (in ~N), theta(angle), lambda - wavelength in sin fact of theta,
                double lambda = j;
                //gamma is the spatial aspect ratio.
                double gamma = 1;
                //psi is the phase offset.

                Mat gaborKernel = getGaborKernel(cv::Size(size, size), sigma, theta, lambda, gamma, 0);
                filter2D(outAsFloat, gaborOut, CV_32F, gaborKernel);

                //get max min to convert back to 0-255 pixel range
                double xmin[4], xmax[4];
                minMaxIdx(gaborOut, xmin, xmax);

                //pixels in range 0-255
                gaborOut.convertTo(gaborImage, CV_8U, 255.0 / (xmax[0] - xmin[0]),
                                   -255 * xmin[0] / (xmax[0] - xmin[0]));
                this->gabor.push_back(gaborImage);
                //imshow("2-CV_8U", gaborImage);
                //waitKey(0);
            }
        }
    }

    void showGabor(){
        //namedWindow("Gabor Window", cv::WINDOW_AUTOSIZE);
        imshow("2-CV_8U", this->gabor.at(1));
        waitKey(0);
    }

private:

    void getClassNumber(){
        int w = -1;
        string b = "";
        stringstream ss(this->filename);
        stringstream out("");
        string token;

        while(std::getline(ss, token, char(47))) {
            b = token;
        }
        b = b.substr(0,3);
        w = stoi(b);

        this->classNo = w;
    }

};

#endif
