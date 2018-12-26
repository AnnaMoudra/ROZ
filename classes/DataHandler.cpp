#include "GaborHandler.cpp"
#include <filesystem>
#include <tchar.h>
#include <iostream>
#include <random>

namespace fs = std::experimental::filesystem;

class DataHandler{
public:
    string path = "";
    string pathOut = "";
    vector<Image *> all_data;
    vector<Image *> training_set;
    vector<Image *> testing_set;

    DataHandler(string p, string o){
        this->path = p;
        this->pathOut = o;
    }

    void load(){
        stringstream input("");
        int data_cnt = 0;
        for (const auto & p : fs::directory_iterator(path)) {
            input.str("");
            input << p;
            string a = replaceBackSlashes(input.str());
            Image * tmp = loadImage(a);
            tmp->name = getImageName(a);
            tmp->out = getImageOut(a);
            tmp->left = getImageLeft(a);
            tmp->extractHistogram();
            tmp->cropImg();
            //this->saveAreaImage(tmp);
            for(auto a: tmp->areas)
                tmp->extractChannels(a);

            for(auto ch: tmp->channels)
                tmp->EF_GABOR(ch, true);

            tmp->extractGaborFeatures();
            this->all_data.push_back(tmp);
            data_cnt++;
            //cout<<"progress: "<<data_cnt<<"/813"<<endl;
        }
        return;
    }

    void clearSets(){
        this->testing_set.clear();
        this->training_set.clear();
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
        else if(ratio == 50){
            //ratio 1:1
            int lastCls = 1;
            vector<Image *> clss;
            for(auto img: this->all_data){
                if(lastCls < img->classNo){
                    this->pickHalfFromClass(clss);
                    clss.clear();
                    lastCls = img->classNo;
                }
                clss.push_back(img);
            }
            pickHalfFromClass(clss);
        }
        else if(ratio == 51){
            //ratio 1:1
            int lastCls = 1;
            vector<Image *> clss;
            for(auto img: this->all_data){
                if(lastCls < img->classNo){
                    this->pickHalfAndSomeFromClass(clss);
                    clss.clear();
                    lastCls = img->classNo;
                }
                clss.push_back(img);
            }
            pickHalfAndSomeFromClass(clss);
        }
        else if(ratio == 70){
            //ratio 30:70
            int lastCls = 1;
            vector<Image *> clss;
            for(auto img: this->all_data){
                if(lastCls < img->classNo){
                    this->pickSeventyFromClass(clss);
                    clss.clear();
                    lastCls = img->classNo;
                }
                clss.push_back(img);
            }
            pickSeventyFromClass(clss);
        }
        //print to file
        //this->printSets();
    }


    void saveAreaImage(Image * a){
        string p = a->name+"_area.jpg";
        cout<<"Saving to: "<<this->pathOut+p<<endl;
        imwrite(this->pathOut+p, a->area);
        return;
    }

    void saveGaborImages(Image * a){
        cout<<"Saving gabor images"<<endl;
        int i = 0;
        for(auto image : a->gabor){
            string p = a->name+"_"+to_string(i)+".jpg";
            cout<<"Saving to: "<<this->pathOut+p<<endl;
            imwrite(this->pathOut+p, image);
            i++;
            cout<<"progress: "<<i<<"/40"<<endl;
        }
        return;
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

    void pickHalfAndSomeFromClass(vector<Image *> clss){
        vector<Image *> in;
        vector<Image *> out;
        int testingCnt = 0;
        for(auto im : clss){
            if(im->out)
                out.push_back(im);
            else
                in.push_back(im);
        }

        int in_half = in.size()/2;
        int out_half = out.size()/2;
        int in_rem = in.size()%2;
        int out_rem = out.size()%2;

        //rozdel in
        for(int i=0; i < in_half; i++){
            std::random_device rd; // obtain a random number from hardware
            std::mt19937 eng(rd()); // seed the generator
            std::uniform_int_distribution<> distr(0, in.size()-1); // define the range
            int idx = distr(eng);
            Image * one  = in.at(idx);
            in.erase(in.begin()+idx);
            this->testing_set.push_back(one);
            testingCnt++;
        }

        //rozdel out
        for(int i=0; i < out_half; i++){
            std::random_device rd; // obtain a random number from hardware
            std::mt19937 eng(rd()); // seed the generator
            std::uniform_int_distribution<> distr(0, out.size()-1); // define the range
            int idx = distr(eng);
            Image * one  = out.at(idx);
            out.erase(out.begin()+idx);
            this->testing_set.push_back(one);
            testingCnt++;
        }

        if(in_rem == 0 && out_rem == 0){
            //zbytek je pulka
            this->training_set.insert(this->training_set.end(), in.begin(), in.end());
            this->training_set.insert(this->training_set.end(), out.begin(), out.end());
        }
        else if(in_rem && out_rem){
            //zbytek jsou 2
            if(testingCnt >= (in.size() + out.size() - 1)){
                this->training_set.insert(this->training_set.end(), in.begin(), in.end());
                this->training_set.insert(this->training_set.end(), out.begin(), out.end());
            }
            else {
                std::random_device rd0; // obtain a random number from hardware
                std::mt19937 eng0(rd0()); // seed the generator
                std::uniform_int_distribution<> distr0(0, 1); // define the range
                int set = distr0(eng0);
                if (set > 0) {
                    std::random_device rd; // obtain a random number from hardware
                    std::mt19937 eng(rd()); // seed the generator
                    std::uniform_int_distribution<> distr(0, out.size() - 1); // define the range
                    int idx = distr(eng);
                    Image *one = out.at(idx);
                    out.erase(out.begin() + idx);
                    this->testing_set.push_back(one);
                    testingCnt++;
                } else {
                    std::random_device rd; // obtain a random number from hardware
                    std::mt19937 eng(rd()); // seed the generator
                    std::uniform_int_distribution<> distr(0, in.size() - 1); // define the range
                    int idx = distr(eng);
                    Image *one = in.at(idx);
                    in.erase(in.begin() + idx);
                    this->testing_set.push_back(one);
                    testingCnt++;
                }
                //zbytek je pulka
                this->training_set.insert(this->training_set.end(), in.begin(), in.end());
                this->training_set.insert(this->training_set.end(), out.begin(), out.end());
            }
        }
        else if(in_rem){
            //zbytek jsou 2
            if(testingCnt >= (in.size() + out.size() - 1)){
                this->training_set.insert(this->training_set.end(), in.begin(), in.end());
                this->training_set.insert(this->training_set.end(), out.begin(), out.end());
            }
            else {
                std::random_device rd; // obtain a random number from hardware
                std::mt19937 eng(rd()); // seed the generator
                std::uniform_int_distribution<> distr(0, in.size() - 1); // define the range
                int idx = distr(eng);
                Image *one = in.at(idx);
                in.erase(in.begin() + idx);
                this->testing_set.push_back(one);

                //zbytek je pulka
                this->training_set.insert(this->training_set.end(), in.begin(), in.end());
                this->training_set.insert(this->training_set.end(), out.begin(), out.end());
            }
        }
        else{
            if(testingCnt >= (in.size() + out.size() - 1)){
                this->training_set.insert(this->training_set.end(), in.begin(), in.end());
                this->training_set.insert(this->training_set.end(), out.begin(), out.end());
            }
            else {
                std::random_device rd; // obtain a random number from hardware
                std::mt19937 eng(rd()); // seed the generator
                std::uniform_int_distribution<> distr(0, out.size() - 1); // define the range
                int idx = distr(eng);
                Image *one = out.at(idx);
                out.erase(out.begin() + idx);
                this->testing_set.push_back(one);
                testingCnt++;
                //zbytek je pulka
                this->training_set.insert(this->training_set.end(), in.begin(), in.end());
                this->training_set.insert(this->training_set.end(), out.begin(), out.end());
            }
        }
    }

    void pickHalfFromClass(vector<Image *> clss){
        vector<Image *> in;
        vector<Image *> out;
        int testingCnt = 0;
        for(auto im : clss){
            if(im->out)
                out.push_back(im);
            else
                in.push_back(im);
        }

        int in_half = in.size()/2;
        int out_half = out.size()/2;
        int in_rem = in.size()%2;
        int out_rem = out.size()%2;

        if(in.size() == 1 && out.size() == 1){
            std::random_device rd0; // obtain a random number from hardware
            std::mt19937 eng0(rd0()); // seed the generator
            std::uniform_int_distribution<> distr0(0, 1); // define the range
            int set = distr0(eng0);
            if (set > 0) {
                this->testing_set.insert(this->testing_set.end(), in.begin(), in.end());
                this->training_set.insert(this->training_set.end(), out.begin(), out.end());
            } else {
                this->testing_set.insert(this->testing_set.end(), out.begin(), out.end());
                this->training_set.insert(this->training_set.end(), in.begin(), in.end());
            }
            return;
        }

        //rozdel in
        for(int i=0; i < in_half; i++){
            std::random_device rd; // obtain a random number from hardware
            std::mt19937 eng(rd()); // seed the generator
            std::uniform_int_distribution<> distr(0, in.size()-1); // define the range
            int idx = distr(eng);
            Image * one  = in.at(idx);
            in.erase(in.begin()+idx);
            this->testing_set.push_back(one);
            testingCnt++;
        }

        //rozdel out
        for(int i=0; i < out_half; i++){
            std::random_device rd; // obtain a random number from hardware
            std::mt19937 eng(rd()); // seed the generator
            std::uniform_int_distribution<> distr(0, out.size()-1); // define the range
            int idx = distr(eng);
            Image * one  = out.at(idx);
            out.erase(out.begin()+idx);
            this->testing_set.push_back(one);
            testingCnt++;
        }

        if(in_rem == 0 && out_rem == 0){
            //zbytek je pulka
            this->training_set.insert(this->training_set.end(), in.begin(), in.end());
            this->training_set.insert(this->training_set.end(), out.begin(), out.end());
        }
        else if(in_rem && out_rem){
            //zbytek jsou 2
            if(testingCnt >= (in.size() + out.size() - 1)){
                this->training_set.insert(this->training_set.end(), in.begin(), in.end());
                this->training_set.insert(this->training_set.end(), out.begin(), out.end());
            }
            else {
                std::random_device rd0; // obtain a random number from hardware
                std::mt19937 eng0(rd0()); // seed the generator
                std::uniform_int_distribution<> distr0(0, 1); // define the range
                int set = distr0(eng0);
                if (set > 0) {
                    std::random_device rd; // obtain a random number from hardware
                    std::mt19937 eng(rd()); // seed the generator
                    std::uniform_int_distribution<> distr(0, out.size() - 1); // define the range
                    int idx = distr(eng);
                    Image *one = out.at(idx);
                    out.erase(out.begin() + idx);
                    this->testing_set.push_back(one);
                    testingCnt++;
                } else {
                    std::random_device rd; // obtain a random number from hardware
                    std::mt19937 eng(rd()); // seed the generator
                    std::uniform_int_distribution<> distr(0, in.size() - 1); // define the range
                    int idx = distr(eng);
                    Image *one = in.at(idx);
                    in.erase(in.begin() + idx);
                    this->testing_set.push_back(one);
                    testingCnt++;
                }
                //zbytek je pulka
                this->training_set.insert(this->training_set.end(), in.begin(), in.end());
                this->training_set.insert(this->training_set.end(), out.begin(), out.end());
            }
        }
        else if(in_rem){
                std::random_device rd; // obtain a random number from hardware
                std::mt19937 eng(rd()); // seed the generator
                std::uniform_int_distribution<> distr(0, in.size() - 1); // define the range
                int idx = distr(eng);
                Image *one = in.at(idx);
                in.erase(in.begin() + idx);
                //zbytek je pulka
                this->training_set.insert(this->training_set.end(), in.begin(), in.end());
                this->training_set.insert(this->training_set.end(), out.begin(), out.end());
        }
        else{
                std::random_device rd; // obtain a random number from hardware
                std::mt19937 eng(rd()); // seed the generator
                std::uniform_int_distribution<> distr(0, out.size() - 1); // define the range
                int idx = distr(eng);
                Image *one = out.at(idx);
                out.erase(out.begin() + idx);
                //zbytek je pulka
                this->training_set.insert(this->training_set.end(), in.begin(), in.end());
                this->training_set.insert(this->training_set.end(), out.begin(), out.end());
        }
    }

    void pickSeventyFromClass(vector<Image *> clss){
        vector<Image *> in;
        vector<Image *> out;
        int testingCnt = 0;
        for(auto im : clss){
            if(im->out)
                out.push_back(im);
            else
                in.push_back(im);
        }

        int in_seventy = floor(in.size()*0.7);
        int out_seventy = floor(out.size()*0.7);
        int in_thirty = floor(in.size()*0.3);
        int out_thirty = floor(out.size()*0.3);
        int in_rem = in.size()-(in_seventy+in_thirty);
        int out_rem = out.size()-(out_seventy+out_thirty);
        /*
        cout<<"before picking"<<endl;
        cout<<"in seventy "<<in_seventy<<endl;
        cout<<"out seventy "<<out_seventy<<endl;
        cout<<"in thirty "<<in_thirty<<endl;
        cout<<"out thirty "<<out_thirty<<endl;
        cout<<"in rem "<<in_rem<<endl;
        cout<<"out rem "<<out_rem<<endl;
         */

        if(in.size() == 1 && out.size() == 1){
            std::random_device rd0; // obtain a random number from hardware
            std::mt19937 eng0(rd0()); // seed the generator
            std::uniform_int_distribution<> distr0(0, 1); // define the range
            int set = distr0(eng0);
            if (set > 0) {
                this->testing_set.insert(this->testing_set.end(), in.begin(), in.end());
                this->training_set.insert(this->training_set.end(), out.begin(), out.end());
            } else {
                this->testing_set.insert(this->testing_set.end(), out.begin(), out.end());
                this->training_set.insert(this->training_set.end(), in.begin(), in.end());
            }
            return;
        }

        if(in.size() == 2 && out.size() == 0){
            std::random_device rd0; // obtain a random number from hardware
            std::mt19937 eng0(rd0()); // seed the generator
            std::uniform_int_distribution<> distr0(0, 1); // define the range
            int set = distr0(eng0);
            if (set > 0) {
                this->testing_set.push_back(in.at(0));
                this->training_set.push_back(in.at(1));
            }
            else{
                this->testing_set.push_back(in.at(1));
                this->training_set.push_back(in.at(0));
            }
            return;
        }

        if(out.size() == 2 && in.size() == 0){
            std::random_device rd0; // obtain a random number from hardware
            std::mt19937 eng0(rd0()); // seed the generator
            std::uniform_int_distribution<> distr0(0, 1); // define the range
            int set = distr0(eng0);
            if (set > 0) {
                this->testing_set.push_back(out.at(0));
                this->training_set.push_back(out.at(1));
            }
            else{
                this->testing_set.push_back(out.at(1));
                this->training_set.push_back(out.at(0));
            }
            return;
        }

        //rozdel in
        for(int i=0; i < in_seventy; i++){
            std::random_device rd; // obtain a random number from hardware
            std::mt19937 eng(rd()); // seed the generator
            std::uniform_int_distribution<> distr(0, in.size()-1); // define the range
            int idx = distr(eng);
            Image * one  = in.at(idx);
            in.erase(in.begin()+idx);
            this->testing_set.push_back(one);
            testingCnt++;
        }

        //rozdel out
        for(int i=0; i < out_seventy; i++){
            std::random_device rd; // obtain a random number from hardware
            std::mt19937 eng(rd()); // seed the generator
            std::uniform_int_distribution<> distr(0, out.size()-1); // define the range
            int idx = distr(eng);
            Image * one  = out.at(idx);
            out.erase(out.begin()+idx);
            this->testing_set.push_back(one);
            testingCnt++;
        }

        if(in_rem == 0 && out_rem == 0){
            this->training_set.insert(this->training_set.end(), in.begin(), in.end());
            this->training_set.insert(this->training_set.end(), out.begin(), out.end());
        }
        else if(in_rem && out_rem){
            std::random_device rd0; // obtain a random number from hardware
            std::mt19937 eng0(rd0()); // seed the generator
            std::uniform_int_distribution<> distr0(0, 1); // define the range
            int set = distr0(eng0);
            if (set > 0) {
                std::random_device rd; // obtain a random number from hardware
                std::mt19937 eng(rd()); // seed the generator
                std::uniform_int_distribution<> distr(0, out.size() - 1); // define the range
                int idx = distr(eng);
                Image *one = out.at(idx);
                out.erase(out.begin() + idx);
                this->testing_set.push_back(one);
                testingCnt++;
            } else {
                std::random_device rd; // obtain a random number from hardware
                std::mt19937 eng(rd()); // seed the generator
                std::uniform_int_distribution<> distr(0, in.size() - 1); // define the range
                int idx = distr(eng);
                Image *one = in.at(idx);
                in.erase(in.begin() + idx);
                this->testing_set.push_back(one);
                testingCnt++;
            }
            //zbytek je pulka
            this->training_set.insert(this->training_set.end(), in.begin(), in.end());
            this->training_set.insert(this->training_set.end(), out.begin(), out.end());
        }
        else if(in_rem){
            for(int i = 0; i < in_rem; i++){
                std::random_device rd; // obtain a random number from hardware
                std::mt19937 eng(rd()); // seed the generator
                std::uniform_int_distribution<> distr(0, in.size() - 1); // define the range
                int idx = distr(eng);
                Image *one = in.at(idx);
                in.erase(in.begin() + idx);
                this->testing_set.push_back(one);
            }
            this->training_set.insert(this->training_set.end(), in.begin(), in.end());
            this->training_set.insert(this->training_set.end(), out.begin(), out.end());
        }
        else{
            for(int i = 0; i < out_rem; i++) {
                std::random_device rd; // obtain a random number from hardware
                std::mt19937 eng(rd()); // seed the generator
                std::uniform_int_distribution<> distr(0, out.size() - 1); // define the range
                int idx = distr(eng);
                Image *one = out.at(idx);
                out.erase(out.begin() + idx);
                this->testing_set.push_back(one);
            }
            //zbytek je pulka
            this->training_set.insert(this->training_set.end(), in.begin(), in.end());
            this->training_set.insert(this->training_set.end(), out.begin(), out.end());
        }
        return;
    }

    void printSets(){
        string pathTest = "./../test.txt";
        string pathTrain = "./../train.txt";
        stringstream out_line;


        ofstream testFile;
        ofstream trainFile;

        testFile.open(pathTest.c_str());
        for(auto im: this->testing_set){
            stringstream  ss(im->filename);
            ss << endl;
            testFile << ss.str();
        }
        testFile.close();

        trainFile.open(pathTrain.c_str());
        for(auto im: this->training_set){
            stringstream  ss(im->filename);
            ss << endl;
            trainFile << ss.str();
        }
        trainFile.close();

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

    string getImageName(string a){
        string b = "";
        stringstream ss(a);
        stringstream out("");
        string token;

        while(std::getline(ss, token, char(47))) {
            out << token << char(47);
        }
        token.pop_back();
        return token;
    }

    bool getImageOut(string a){
        bool outF = false;
        stringstream ss(a);
        string token;

        int i = 0;
        while(std::getline(ss, token, char(95))) {
            if(i == 3){
                if(token.compare("OU") == 0){
                    //cout<<a<<" is OUT"<<endl;
                    outF = true;
                    break;
                }
            }
            i++;
        }

        return outF;
    }

    bool getImageLeft(string a){
        bool leftF = false;
        stringstream ss(a);
        string token;

        int i = 0;
        while(std::getline(ss, token, char(95))) {
            if(i == 4){
                if(token.compare("F") == 0){
                    //cout<<a<<" is LEFT"<<endl;
                    leftF = true;
                    break;
                }
            }
            i++;
        }

        return leftF;
    }

    Image * loadImage(string path){
        Image * im_new = new Image(path);
        im_new->loadImage();
        return im_new;
    }

};