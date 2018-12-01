#include "Image.cpp"

using namespace cv;
using namespace std;


class FeatureHandler{

public:
    Image * img1;
    Image * img2;
    vector<pair<float, Image *>> histIntersections;

    FeatureHandler(){}

    void SetImgs(Image * i, Image * b){
        this->img1 = i;
        this->img2 = b;
    }

    void testSet(){
        img1->EF_ORB();
        // Match features.
        std::vector<cv::DMatch> matches;
        matchFeatures(img1->descriptors, img2->descriptors, matches);
        // Draw matches.
        Mat image_matches;
        drawMatches(img1->img, img1->keypoints, img2->img, img2->keypoints, matches, image_matches);

        namedWindow("Image Window", cv::WINDOW_AUTOSIZE);
        imshow("Image Window", image_matches);
        waitKey(0);
    }

    #define RATIO    0.75
    void matchFeatures(const cv::Mat &query, const cv::Mat &target, std::vector<cv::DMatch> &goodMatches) {
        std::vector<std::vector<cv::DMatch>> matches;
        FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));;
        // Find 2 best matches for each descriptor to make later the second neighbor test.
        matcher.knnMatch(query, target, matches, 2);
        // Second neighbor ratio test.
        for (unsigned int i = 0; i < matches.size(); ++i) {
            if (matches[i][0].distance < matches[i][1].distance * RATIO)
                goodMatches.push_back(matches[i][0]);
        }
    }


    void histogramComparison(){
        float intersection  = this->compareHistIntersection(this->img1->histogram, this->img2->histogram);

        this->histIntersections.push_back(make_pair(intersection, img2));
    }

    Image * getMaxIntersection(){
        float maxIntersection = 0.0;
        string bestFile= "";
        Image * bestImg;
        int w = -1;
        for(auto a: histIntersections){
            float b_float = a.first;
            if((b_float - maxIntersection) > 10e-6){
                maxIntersection = b_float;
                bestImg = a.second;
                bestFile = a.second->filename;
                w = a.second->classNo;
            }
        }

        /*
        cout<<"Best result:"<<endl;
        cout<<"\tFile: "<< bestImg->filename<<endl;
        cout<<"\tIntersection: "<<maxIntersection<<endl;
        cout<<"\tClass: "<<w<<endl;
         */
        return bestImg;
    }






private:
    float compareHistIntersection(MatND hist1, MatND hist2){
        float intersection = 0.0;
        for( int i = 0; i < 4; i++ )
        { int compare_method = i;
            double hist1_hist1 = compareHist( hist1, hist1, compare_method );
            double hist1_hist2 = compareHist( hist1, hist2, compare_method );
            //cout<<"Method type: "<<i<<endl;
            //cout<<"\tH1 - H1: "<<hist1_hist1<<endl;
            //cout<<"\t\tH1 - H2: "<<hist1_hist2<<endl;
            if(i==2){
                intersection = hist1_hist2;
            }

        }
        return intersection;
    }

};
