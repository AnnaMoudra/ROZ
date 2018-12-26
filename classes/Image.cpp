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
    vector<Mat> channels;
    vector<Mat> areas;
    vector<KeyPoint> keypoints;
    Mat descriptors;
    int classNo = -1;
    //LAB histogram
    MatND histogram;
    //selected gabor features
    vector<Mat> gabor;
    //extracted gabor feature vector
    vector<int> gaborFeatures;

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

    void preProcessing(){
        //TODO

    }

    void extractChannels(Mat im){
        //cout<<im.channels()<<endl;
        vector<Mat> channels (im.channels());
        split(im, channels);
        for(auto m: channels)
            this->channels.push_back(m);

    }

    //todo EXTRACT ALL ??
    void extractGaborFeatures(){
        //8 rotations 1 setup = 8*1 response matrices
        vector<int> loc_energy;
        //vector<int> mean_amplitude;
        for(auto g : this->gabor){
            //where g is one Gabor-filtered image (img->img)
            // g is a response matrice - My image convolved with Gabor feature (gabor filter)
            // feature vector: Local Energy,Mean Amplitude,Phase Amlitude or Orientation whose local has maximumEnergy

            //Local Energy = summing up the squared value of each matrix value from a response matrix
            int sum = 0;
            for(int i=0; i<g.rows; i++) {
                for (int j = 0; j < g.cols; j++) {
                    int pixel = g.at<uchar>(i, j);
                    int pixel_p2 = pixel * pixel;
                    sum += pixel;
                }
            }
            loc_energy.push_back(sum);
            /*
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
            */
        }
        vector<int> all;
        // preallocate memory
        all.reserve( loc_energy.size() ); //+ mean_amplitude.size() );
        all.insert( all.end(), loc_energy.begin(), loc_energy.end() );
        //all.insert( all.end(), mean_amplitude.begin(), mean_amplitude.end() );

        //return ft vector back ti Image
        this->gaborFeatures = all;
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

        calcHist( &hsv_base, 1, channels, Mat(), this->histogram, 2, histSize, ranges, true, false );
        normalize( this->histogram, this->histogram, 0, 1, NORM_MINMAX, -1, Mat() );
    }

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
        //pixel in range 0.-1.
        outImg.convertTo(outAsFloat, CV_32F);

        for (int j = 3; j <= 13; j += 2) {
            for (int i = 0; i <= 120; i += 30) {
                Mat gaborImage, gaborImageSin, gaborImageCos;
                //kernel size and sigma (in ~N(mi, sigma))
                //scale for window 30x35
                int size = 14;
                int sigma = 7;
                //theta: angle of rotation
                double theta = i;
                //lambda - wavelength in sin fact of theta,
                double lambda = j;
                //gamma is the spatial aspect ratio.
                double gamma = 0.25;

                //psi is the phase offset. (cos: psi=0, sin: psi=pi/2)
                Mat gaborKernelCos = getGaborKernel(cv::Size(size, size), sigma, theta, lambda, gamma, 0);
                Mat gaborKernelSin = getGaborKernel(cv::Size(size, size), sigma, theta, lambda, gamma, M_PI/2.0);

                //filter2D(outAsFloat, gaborOut, CV_32F, gaborKernel);
                Mat sin_response, cos_response, temp, image_out;
                filter2D(outAsFloat,sin_response, CV_32F, gaborKernelSin, cv::Point(-1,-1));
                filter2D(outAsFloat,cos_response, CV_32F, gaborKernelCos, cv::Point(-1,-1));
                //calculate Energy
                multiply(sin_response, sin_response, sin_response);
                multiply(cos_response, cos_response, cos_response);
                //get max min to convert back to 0-255 pixel range
                double xmin[4], xmax[4];
                //minMaxIdx(gaborOut, xmin, xmax);
                minMaxIdx(sin_response, xmin, xmax);
                minMaxIdx(cos_response, xmin, xmax);
                //pixels in range 0-255
                sin_response.convertTo(gaborImageSin, CV_8U, 255.0 / (xmax[0] - xmin[0]),-255 * xmin[0] / (xmax[0] - xmin[0]));
                cos_response.convertTo(gaborImageCos, CV_8U, 255.0 / (xmax[0] - xmin[0]),-255 * xmin[0] / (xmax[0] - xmin[0]));

                this->gabor.push_back(gaborImageSin);
                this->gabor.push_back(gaborImageCos);
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
