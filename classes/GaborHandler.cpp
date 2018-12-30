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

    //unused
    int hammingDistance(int x, int y) {
        int z  = x ^ y;
        int r = 0;
        for (; z > 0; z >>= 1) {
            r += z & 1;
        }
        return r;
    }

    //unused
    int fitHD(Image * a){
        //cout<<"Fitting image HD"<<endl;
        vector<int> BV_X = a->gaborFeatures;
        unsigned long long best = 0;
        int win = -1;

        for(auto person: trainSet){
            for(int i = 0; i < person->features.size(); i++){
                unsigned long long HM_D = 0;
                for(int k = 0; k < person->features.at(i).size(); k++){
                    if(BV_X.size() <= k){
                        cout<<"BV_X is too short"<<endl;
                    }
                    else{
                        //todo hamming distance
                        unsigned long ixor = this->hammingDistance(BV_X.at(k), person->features.at(i).at(k));
                        HM_D += ixor;
                    }
                }

                HM_D = HM_D / person->features.at(i).size();
                if(best == 0 || HM_D < best){
                    best = HM_D;
                    win = person->c;
                    cout<<"NEW BEST: "<<win<<endl;
                }
            }
        }
        cout<<"Class res: "<<win<<endl;
        cout<<"HD best: "<<best<<endl;
        if(win == a->classNo){
            cout<<"CORRECT!"<<endl;
            return 1;
        }
        else{
            cout<<"WRONG CLASS! should be "<<a->classNo<<endl;
            return 0;
        }
    }

    //unused
    int fitXOR(Image * a){
        //cout<<"Fitting image HD"<<endl;
        vector<int> BV_X = a->gaborFeatures;
        long long best = 0;
        int win = -1;

        for(auto person: trainSet){
            for(int i = 0; i < person->features.size(); i++){
                long long X_D = 0;
                for(int k = 0; k < person->features.at(i).size(); k++){
                    if(BV_X.size() <= k){
                        cout<<"BV_X is too short"<<endl;
                    }
                    else{
                        long int ixor = BV_X.at(k) ^ person->features.at(i).at(k);
                        X_D += ixor;
                    }
                }

                X_D = X_D / person->features.at(i).size();
                if(best == 0 || X_D < best){
                    best = X_D;
                    win = person->c;
                    cout<<"NEW BEST: "<<win<<endl;
                }
            }
        }
        cout<<"Class res: "<<win<<endl;
        cout<<"XOR best: "<<best<<endl;
        if(win == a->classNo){
            cout<<"CORRECT!"<<endl;
            return 1;
        }
        else{
            cout<<"WRONG CLASS! should be "<<a->classNo<<endl;
            return 0;
        }
    }


    /* fit image against train data using Euclidian distance*/
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

    /* fit image against train data using Euclidian distance*/
    int fitEDdist(Image * a){
        //cout<<"Fitting image HD"<<endl;
        vector<double> BV_X = a->gaborFeaturesD;
        long double  best = 0;
        int win = -1;

        for(auto person: trainSet){
            //cout<<"New person ("<<person->c<<"), ft size: "<<person->features.size()<<endl;
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

    //unused
     void trainAllVsAll(){
        cout<<"Training ALL classes at once"<<endl;
        this->AllVsAll(this->trainSet);
        cout<<"Training finished"<<endl;
    }

    //unused
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

    //unused
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

    void getStandardValues(vector<Image *> data){
        //standard mean

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



    //unused
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

    //unused
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
        this->svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 3000, 1e-6));
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


    //unused
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

