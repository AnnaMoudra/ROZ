//50 by 50
//n-1 by 1
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <istream>
#include <tchar.h>
#include <stdio.h>

#include "classes/Image.cpp"

//testing 5
using namespace std;
namespace fs = std::experimental::filesystem;

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
        i->EF();
    }
}

int main(){

    string path = "./../test_5";
    vector <Image *> training_set;
    int data_cnt = 0;
    stringstream input("");
    Image * test;
    for (const auto & p : fs::directory_iterator(path)) {
        input.str("");
        input << p;
        string a = replaceBackSlashes(input.str());
        training_set.push_back(loadImage(a));
        data_cnt++;
        if(data_cnt == 10){
            test = training_set.back();
            training_set.pop_back();
        }
    }
    data_cnt--;
    cout << "Number of files to be trained: " << data_cnt << endl;

    trainSet(training_set);

    test->showImg();
    test->imgToLAB();
    test->showImg();
    return 0;
}

