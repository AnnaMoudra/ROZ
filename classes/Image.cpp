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
    string name;
    bool out;
    bool left;
    Mat original;
    Mat img;
    Mat area;
    vector<Mat> channels; //BGR color channels
    vector<Mat> areas; // areas used for gabor filtering feature selection
    vector<KeyPoint> keypoints; //unused
    Mat descriptors;
    int classNo = -1;
    //HSV histogram
    MatND histogram;
    //selected gabor responses
    vector<Mat> gabor;
    //extracted gabor feature vector
    vector<int> gaborFeatures;
    vector<double> gaborFeaturesD;

    Image(string fn){
        this->filename = fn.c_str();
        getClassNumber();
    }

    void loadImage(){
        this->img = imread(this->filename, CV_LOAD_IMAGE_COLOR);
        this->img.copyTo(this->original);
        // Read in the image file
        int rows = this->img.rows;
        int cols = this->img.cols;
        int channels = this->img.channels();
    }

    void saveImage(string p){
        string p2 = p+this->name+".jpg";
        imwrite(p2, this->original);
        return;
    }

    /* Extract areas used for feature selection */
    void cropImg(){
        Rect myROI(250, 10, 100, 70);
        vector<Rect> ars;

        int max = 28; //28
        int half = max/2;
        int width = 30;
        int height = 35;
        int w_offset = (600 - (half*width))/2;

        for(int i = 1; i <= max; i++){
            if(i <= half ){
                Rect rec(w_offset + ((i-1)*width), 0, width, height);
                ars.push_back(rec);
            }
            else{
                Rect rec2(w_offset + (((i-half)-1)*width), 0 + height, width, height);
                ars.push_back(rec2);
            }
        }

        this->original.copyTo(this->area);
        //cout<<"Area count: "<<ars.size()<<endl;
        for(auto area: ars){
            Mat tmp;
            this->original.copyTo(tmp);
            tmp = tmp(area);
            this->areas.push_back(tmp);
            rectangle(this->area, area, cv::Scalar(0, 255, 0), 2);
        }
        return;
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


    void extractChannels(Mat im){
        //cout<<im.channels()<<endl;
        vector<Mat> channels (im.channels());
        split(im, channels);
        for(auto m: channels)
            this->channels.push_back(m);

    }

    //todo EXTRACT ALL
    void extractGaborFeatures(){
        vector<int> loc_energy;
        vector<double> doubleFeatures;


        for(int k=0; k < this->gabor.size(); k = k+2){

            Mat g1, g2;
            g1 = this->gabor.at(k);
            g2 = this->gabor.at(k+1);
            int sum1 = 0;
            int sum2 = 0;
            int sigma1 = 0;
            int sigma2 = 0;
            Scalar mean1, mean2, stddev1, stddev2;

            for(int i=0; i<g1.rows; i++) {
                for (int j = 0; j < g1.cols; j++) {

                    int pixel1 = g1.at<uchar>(i, j);
                    int pixel2 = g2.at<uchar>(i, j);
                     //toto funguje cca o 3 % hure nez pouhy soucet ctvercu kazdeho pixelu
                    //sum += sqrt(pixel1 + pixel2);
                    sum1 += pixel1*pixel1;
                    sum2 += pixel2*pixel2;

                }
            }
            meanStdDev( g1, mean1, stddev1 );
            meanStdDev( g2, mean2, stddev2 );
            //loc_energy.push_back(sum);
            loc_energy.push_back(sum1);
            loc_energy.push_back(sum2);

            doubleFeatures.push_back((double)mean1[0]);
            doubleFeatures.push_back((double)stddev1[0]);
            doubleFeatures.push_back((double)mean2[0]);
            doubleFeatures.push_back((double)stddev2[0]);
        }
        //vector<int> all;
        //all.reserve( loc_energy.size());
        //all.insert( all.end(), loc_energy.begin(), loc_energy.end() );

        //return ft vector back ti Image
        this->gaborFeatures = loc_energy;
        this->gaborFeaturesD = doubleFeatures;
    }

    void extractHistogram(){
        Mat hsv_base;
        Mat tmp;
        this->original.copyTo(tmp);
        //crop
        Rect myROI(240, 0, 120, 70); //56.4
        tmp = tmp(myROI);
        //convert to HSV
        cvtColor(tmp, hsv_base, COLOR_BGR2HSV);

        int h_bins = 50; int s_bins = 60;
        int histSize[] = { h_bins, s_bins };
        float h_ranges[] = { 0, 180 };
        float s_ranges[] = { 0, 256 };
        const float* ranges[] = { h_ranges, s_ranges };
        int channels[] = { 0, 1 };

        //create and save HSV histogram
        calcHist( &hsv_base, 1, channels, Mat(), this->histogram, 2, histSize, ranges, true, false );
        normalize( this->histogram, this->histogram, 0, 1, NORM_MINMAX, -1, Mat() );
    }

    /*
     * Gabor kernels and filtering Gabor responses
     * Saving already squared responses to vector
     */
    void EF_GABOR(Mat im, bool channel) {
        Mat gaborOut, outAsFloat;
        Mat tmp, outImg;
        im.copyTo(tmp);

        //get grayscale
        if(channel == false){
            cv::cvtColor(tmp, outImg, CV_BGR2GRAY);
        }
        else{
            tmp.copyTo(outImg);
        }
        //set pixel to range 0.-1.
        outImg.convertTo(outAsFloat, CV_32F);

        for (int j = 3; j <= 13; j += 2) {
            for (int i = 0; i <= 150; i += 30) {
                Mat gaborImage, gaborImageSin, gaborImageCos;
                //kernel size and sigma (in ~N(mi, sigma))
                //scale for window 30x35
                //lambda - wavelength in sin fact of theta,
                double lambda = j;
                int size = 14;
                int sigma = 0.56*lambda;
                //theta: angle of rotation
                double theta = i;
                //gamma is the spatial aspect ratio.
                double gamma = 0.5;

                //psi is the phase offset. (cos: psi=0, sin: psi=pi/2)
                //real part, symmetric part
                Mat gaborKernelCos = getGaborKernel(cv::Size(size, size), sigma, theta, lambda, gamma, 0);
                //imaginary part
                Mat gaborKernelSin = getGaborKernel(cv::Size(size, size), sigma, theta, lambda, gamma, M_PI/2.0);


                Mat sin_response, cos_response, temp, image_out;
                filter2D(outAsFloat,sin_response, CV_32F, gaborKernelSin, cv::Point(-1,-1));
                filter2D(outAsFloat,cos_response, CV_32F, gaborKernelCos, cv::Point(-1,-1));
                //calculate Energy
                //multiply(sin_response, sin_response, sin_response);
                //multiply(cos_response, cos_response, cos_response);

                //get max min to convert back to 0-255 pixel range
                double xmin[4], xmax[4];
                //minMaxIdx(gaborOut, xmin, xmax);
                minMaxIdx(sin_response, xmin, xmax);
                minMaxIdx(cos_response, xmin, xmax);
                //pixels in range 0-255
                sin_response.convertTo(gaborImageSin, CV_8U, 255.0 / (xmax[0] - xmin[0]),-255 * xmin[0] / (xmax[0] - xmin[0]));
                cos_response.convertTo(gaborImageCos, CV_8U, 255.0 / (xmax[0] - xmin[0]),-255 * xmin[0] / (xmax[0] - xmin[0]));


                this->gabor.push_back(gaborImageSin);
                this->gabor.push_back(gaborImageCos); //real part
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
