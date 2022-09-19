#include "Image.cpp"

using namespace cv;
using namespace std;

class HistogramHandler{
    public:
        Image * img1;
        Image * img2;
        vector<pair<float, Image *>> histIntersections;

        HistogramHandler(){}

        void SetImgs(Image * i, Image * b){
            this->img1 = i;
            this->img2 = b;
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
            return bestImg;
        }

    private:
        float compareHistIntersection(MatND hist1, MatND hist2){
            float intersection = 0.0;
            for( int i = 0; i < 4; i++ ){ 
                int compare_method = i;
                double hist1_hist1 = compareHist( hist1, hist1, compare_method );
                double hist1_hist2 = compareHist( hist1, hist2, compare_method );
                if(i==2){
                    intersection = hist1_hist2;
                }
            }
            return intersection;
        }
};
