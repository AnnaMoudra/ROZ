#include "FeatureHandler.cpp"
#include <map>
#include <list>

using namespace cv;
using namespace std;
using namespace cv::ml;

class Person{
public:
    int c = 0;
    vector<vector<int>> features;

    Person(int k){
        this->c = k;
    }

    void addVector(vector<int> f){
        vector<int> ft(f);
        this->features.push_back(ft);
    }

};


class GaborHandler {
    vector< Person *> trainSet;
    vector<int> test;
    Ptr<SVM> svm;
public:

    GaborHandler(){}



    void clear(){
        this->trainSet.clear();
        Ptr<SVM> svm = SVM::create();
    }

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
            p->addVector(ft);
            if(lastClass < clss){
                this->trainSet.push_back(p);
                lastClass =  clss;
            }
        }
        cout<<"preparation done"<<endl;
    }

     void trainAllVsAll(){
        cout<<"Training ALL classes at once"<<endl;
        this->AllVsAll(this->trainSet);
        cout<<"Training finished"<<endl;
    }



    int fit(Image * a){
        cout<<"Fitting image"<<endl;

        pair <float, float> res = this->fitImage(a);
        cout<<"Class res: "<<res.first<<endl;
        cout<<"Conf: "<<(1.0 / (1.0 + exp(-res.second)))<<endl;
        if(res.first == a->classNo){
            cout<<"CORRECT!"<<endl;
            return 1;
        }
        else{
            return 0;
        }
    }

    int trainOneVsOne(Image * a){
        int ok = 0;

        cout<<"Testing image: "<<a->classNo<<endl;
        vector<double> results;
        //for each Person do OneVsRest
        for(int m = 1; m <= this->trainSet.size(); m++){
            Person * one = this->trainSet.at(m);
            vector <Person *> rest (this->trainSet);
            rest.erase(rest.begin()+m);

            this->OneVsRest(one, rest);
        }
        return ok;
    }

private:

    float** setupHMM(vector<vector<int> > &vals, int N, int M)
    {
        float ** temp;
        temp = new float*[N];
        for(unsigned i=0; (i < N); i++)
        {
            temp[i] = new float[M];
            for(unsigned j=0; (j < M); j++)
            {
                temp[i][j] = vals[i][j];
            }
        }
        return temp;
    }

    float * setupHM(vector<int> &vals, int N) {
        float *temp;
        temp = new float[N];
        for (unsigned i = 0; (i < N); i++) {
            temp[i] = (float) vals[i];
        }
        return temp;
    }

    int * setupHMInt(vector<int> &vals, int N)
    {
        int * temp;
        temp = new int[N];
        for(unsigned i=0; (i < N); i++)
        {
            temp[i] = (int) vals[i];
        }
        return temp;
    }




    void OneVsRest(Person * one, vector<Person *> rest){
        vector<vector<int>> all;
        vector<vector<int>> rest_f;
        vector<int> labels;
        for(auto f : one->features){
            //per sample in person
            labels.push_back(one->c);
            all.push_back(f);
        }
        for(auto p : rest){
            for(auto f : p->features){
                //per sample in person
                labels.push_back(-1);
                all.push_back(f);
            }
        }

        cout<<"About to get float arrays."<<endl;
        int* labels_dat = setupHMInt(labels, labels.size());
        cout<<"Labels ok"<<endl;
        cout<<"all.size(): "<<all.size()<<endl;
        cout<<"all.at(1).size(): "<<all.at(1).size()<<endl;
        Person * tmp = this->trainSet.at(1);
        vector<vector<int>> person_ft (tmp->features);
        vector<int> one_ftv(person_ft.at(1));
        cout<<"trainigSet person ft size: "<<one_ftv.size()<<endl;
        float ** data  = setupHMM(all, (int) all.size(), (int) all.at(1).size());
        cout<<"Float array OK."<<endl;
        Mat trainingDataMat(all.size(), all.at(1).size(), CV_32FC1, data);
        Mat labelsMat(all.size(), 1, CV_32SC1, labels_dat);

        cout<<"Data in Mats."<<endl;
        // Train the SVM
        this->svm = SVM::create();
        //for N >= 2
        this->svm->setType(SVM::ONE_CLASS);
        this->svm->setKernel(SVM::RBF);
        //this->svm->setKernel(SVM::LINEAR);
        this->svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));
        try{
            cout<<"Training starts."<<endl;
            this->svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
        }
        catch(Exception e){
            cout<<"Something happened."<<endl;
            cout<<e.err<<endl;
        }

        cout<<this->svm->isTrained()<<endl;
    }

    void AllVsAll(vector<Person *> dat){
        vector<vector<int>> all;
        vector<int> labels;
        for(auto p : dat){
            cout<<"class: "<<p->c<<" featureVecsCnt:"<<p->features.size()<<endl;
            for(auto f : p->features){
                //per sample in person
                int clss = p->c;
                labels.push_back(clss);
                vector<int> ft(f);
                all.push_back(ft);
            }
        }

        cout<<"About to get float arrays."<<endl;
        int* labels_dat = setupHMInt(labels, labels.size());
        cout<<"Labels ok"<<endl;
        cout<<"all.size(): "<<all.size()<<endl;
        cout<<"all.at(1).size(): "<<all.at(1).size()<<endl;
        Person * tmp = this->trainSet.at(1);
        vector<vector<int>> person_ft (tmp->features);
        vector<int> one_ftv(person_ft.at(1));
        cout<<"trainigSet person ft size: "<<one_ftv.size()<<endl;
        float ** data  = setupHMM(all, (int) all.size(), (int) all.at(1).size());
        cout<<"Float array OK."<<endl;
        Mat trainingDataMat(all.size(), all.at(1).size(), CV_32FC1, data);
        Mat labelsMat(all.size(), 1, CV_32SC1, labels_dat);

        cout<<"Data in Mats."<<endl;
        // Train the SVM
        this->svm = SVM::create();
        //for N >= 2
        this->svm->setType(SVM::C_SVC);
        this->svm->setKernel(SVM::RBF);
        //this->svm->setKernel(SVM::LINEAR);
        this->svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
        try{
            cout<<"Training starts."<<endl;
            this->svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);
        }
        catch(Exception e){
            cout<<"Something happened."<<endl;
            cout<<e.err<<endl;
        }

        cout<<this->svm->isTrained()<<endl;
    }



    pair<float, float> fitImage(Image * a){
        //assert(this->svm != NULL && this->svm->isTrained());
        if(!this->svm->isTrained()){
            cout<<"Model is NOT trained!"<<endl;
        }
        float* sample = setupHM(a->gaborFeatures, a->gaborFeatures.size());
        Mat sampleMat(1, a->gaborFeatures.size(), CV_32F, sample );

        cv::Mat result;
        float  response = 0;
        this->svm->predict(sampleMat, result, cv::ml::StatModel::Flags::RAW_OUTPUT);
        response = this->svm->predict(sampleMat);
        float dist = result.at<float>(0, 0);

        return make_pair(response, dist);
    }


};

