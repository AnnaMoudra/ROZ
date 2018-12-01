#include "GaborHandler.cpp"

#include <filesystem>
#include <tchar.h>
#include <iostream>
#include <random>



namespace fs = std::experimental::filesystem;

class DataHandler{
public:
    string path = "";
    vector<Image *> all_data;
    vector<Image *> training_set;
    vector<Image *> testing_set;

    DataHandler(string p){
        this->path = p;
    }

    void load(){
        stringstream input("");
        int data_cnt = 0;
        for (const auto & p : fs::directory_iterator(path)) {
            input.str("");
            input << p;
            string a = replaceBackSlashes(input.str());
            Image * tmp = loadImage(a);
            //tmp->cropImg();
            //tmp->extractHistogram();
            tmp->EF_GABOR();
            tmp->extractGaborFeatures(tmp);
            this->all_data.push_back(tmp);
            data_cnt++;
        }
    }

    //ratio 1: N-1, 1:1, 30:70
    void extractSets(int ratio){
        if(ratio == 1){
            //prvni
            int lastCls = 1;
            vector<Image *> clss;
            for(auto img: this->all_data){
                if(lastCls < img->classNo){
                    this->pickOneFromClass(clss);
                    clss.clear();
                    lastCls = img->classNo;
                }
                clss.push_back(img);
            }
            pickOneFromClass(clss);

        }
    }

private:

    void pickOneFromClass(vector<Image *> clss){
        int max  = clss.size();
        std::random_device rd; // obtain a random number from hardware
        std::mt19937 eng(rd()); // seed the generator
        std::uniform_int_distribution<> distr(0, max-1); // define the range

        int idx = distr(eng);
        Image * one  = clss.at(idx);
        clss.erase(clss.begin()+idx);

        this->testing_set.push_back(one);
        this->training_set.insert(this->training_set.end(), clss.begin(), clss.end());
    }

    string replaceBackSlashes(string a){
        string b = "";
        stringstream ss(a);
        stringstream out("");
        string token;

        while(std::getline(ss, token, char(92))) {
            out << token << char(47);
        }
        b = out.str();
        b.pop_back();

        return b;
    }

    Image * loadImage(string path){
        Image * im_new = new Image(path);
        im_new->loadImage();
        return im_new;
    }

    void trainSet(vector<Image *> data){
        for(auto i: data){
            i->EF_ORB();
        }
    }
};