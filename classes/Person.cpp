//
// Created by Anna on 11/14/2018.
//
#include "FeatureHandler.cpp"
#include <vector>
#include <string>

using namespace std;

class Person{
public:
    int classNo;
    vector<FeatureHandler> data;

    Person(int k){
        this.classNo = k;
    }

    addData(Image img){

        FeatureHandler * fv = new FeatureHandler(img);
        fv.extract();

        data.push(fv);

    }

    printData(){
        for(auto a: data){

        }
    }

};
