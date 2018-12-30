#include "HistogramHandler.cpp"
#include <map>
#include <list>

using namespace cv;
using namespace std;
using namespace cv::ml;

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


class GaborHandler {
    vector< Person *> trainSet;
    vector<int> test;
    Ptr<SVM> svm;
    double a_mean = 0;
    double a_std = 0;
public:

    GaborHandler(){}


    void clear(){
        this->trainSet.clear();
        Ptr<SVM> svm = SVM::create();
        this->a_mean = 0;
        this->a_std = 0;
    }

    // labeling training data
    void fill(vector<Image *> data){
        int lastClass = -1;
        Person * p;
        for(auto i: data){
            int clss = i->classNo;
            if(lastClass < clss){
                cout<<"new class:"<<clss<<endl;
                p = new Person(clss);
            }
            vector<int> ft(i->gaborFeatures);
            vector<double> ftD(i->gaborFeaturesD);
            p->addVector(ft, ftD);
            if(lastClass < clss){
                this->trainSet.push_back(p);
                lastClass =  clss;
            }
        }
        this->getStandardValues(data);
        cout<<"preparation done"<<endl;
    }


    /* fit image against train data using Euclidian distance */
    int fitED(Image * a){
        //cout<<"Fitting image HD"<<endl;
        vector<int> BV_X = a->gaborFeatures;
        unsigned long long best = 0;
        int win = -1;

        for(auto person: trainSet){
            //cout<<"New person ("<<person->c<<"), ft size: "<<person->features.size()<<endl;
            for(int i = 0; i < person->features.size(); i++){
                unsigned long long ED_D = 0;
                for(int k = 0; k < person->features.at(i).size(); k++){
                    if(BV_X.size() <= k){
                        cout<<"BV_X is too short"<<endl;
                    }
                    else{
                        unsigned long long ed =  pow(BV_X.at(k) - person->features.at(i).at(k),2);
                        ED_D += ed;
                        //cout<<"Final ED_D: "<<ED_D<<endl;
                    }
                }
                if(best == 0 || ED_D < best){
                    best = ED_D;
                    win = person->c;
                    cout<<"NEW BEST ED: "<<win<<endl;
                }
            }
        }
        cout<<"Class res: "<<win<<endl;
        cout<<"ED best: "<<best<<endl;
        if(win == a->classNo){
            cout<<"CORRECT!"<<endl;
            return 1;
        }
        else{
            cout<<"WRONG CLASS! should be "<<a->classNo<<endl;
            return 0;
        }
    }

    /* fit image against train data using normalized absolute distance */
    int fitDist(Image * a){
        vector<double> BV_X = a->gaborFeaturesD;
        long double  best = 0;
        int win = -1;

        for(auto person: trainSet){
            for(int i = 0; i < person->featuresD.size(); i++){
                long double ED_D = 0;
                for(int k = 0; k < person->featuresD.at(i).size(); k+=2){
                    if(BV_X.size() <= k){
                        cout<<"BV_X is too short"<<endl;
                    }
                    else{
                        double mean1 = BV_X.at(k);
                        double std1 = BV_X.at(k+1);
                        double mean_t = person->featuresD.at(i).at(k);
                        double std_t = person->featuresD.at(i).at(k+1);

                        long double ed =  ((abs(mean1-mean_t))/this->a_mean)+((abs(std1-std_t))/this->a_std);
                        ED_D += ed;
                    }
                }
                if(best == 0 || (best - ED_D) > 10e-6){
                    best = ED_D;
                    win = person->c;
                    cout<<"NEW BEST ED: "<<win<<endl;
                }
            }
        }
        cout<<"Class res: "<<win<<endl;
        cout<<"ED best: "<<best<<endl;
        if(win == a->classNo){
            cout<<"CORRECT!"<<endl;
            return 1;
        }
        else{
            cout<<"WRONG CLASS! should be "<<a->classNo<<endl;
            return 0;
        }
    }


private:

    /*
     * Extracts average mean and std from the train data gabor space
     *
     */
    void getStandardValues(vector<Image *> data){
        for(auto m: data){
            for(auto g: m->gabor){ //g je Mat
                Scalar mean, std;
                meanStdDev(g, mean, std);
                this->a_mean += mean[0];
                this->a_std +=std[0];
            }
        }

        this->a_mean /= data.size();
        this->a_std /= data.size();

    }

};

