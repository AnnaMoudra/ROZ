#include <./../opencv/core.hpp>
#include <./../opencv2/opencv.hpp>
#include <iostream>
#include "Image.cpp"

using namespace cv;
using namespace std;


class FeatureHandler(){

public:

    Image img;
    vector<int> features;

    FeatureHandler(Image i){
        this.img = i;
    }

    extract(){
        int meanMiddle = img.getTestValue();
        features.push(meanMiddle);
    }

}
