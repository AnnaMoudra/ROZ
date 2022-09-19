#include "HistogramHandler.cpp"
#include <map>
#include <list>

using namespace std;

class Person{
    public:
        int c = 0;
        vector<vector<int>> features;
        vector<vector<double>> featuresD;

        Person(int k){
            this->c = k;
        }

        void addVector(vector<int> f, vector<double> fd){
            vector<int> ft(f);
            vector<double> ftD(fd);
            this->features.push_back(ft);
            this->featuresD.push_back(ftD);
        }
};