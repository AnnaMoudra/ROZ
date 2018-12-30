// in build: cmake --build . --config Release
// in SRC: .\build\Release\miroz.exe

#include "classes/DataHandler.cpp"
#include <opencv2/opencv.hpp>

//testing 5
using namespace std;

//return 0/1
int simpleHistogram(vector<Image *> data, Image * test){
    FeatureHandler * fh = new FeatureHandler();
    for(auto i: data){
        fh->SetImgs(test, i);
        fh->histogramComparison();
    }

    Image * imageChosen = fh->getMaxIntersection();
    if(test->classNo ==  imageChosen->classNo){
        return 1;
    }
    cout<<"WRONG CLASS"<<endl;
    cout<<"IMG compared: "<<test->filename<<endl;
    cout<<"Class compared: "<<test->classNo<<endl;
    cout<<"Class Chosen: "<<imageChosen->classNo <<endl;
    cout<<"compared: "<<imageChosen->filename<<endl;
    cout<<endl;
    return 0;
}

int main(){
    //string path = "./../test_5";
    string pathGabor = "./../gabor/";
    string pathWrong = "./../wrong/";
    string pathRight = "./../right/";
    string path = "./../iris_all";

    GaborHandler * gh = new GaborHandler();
    DataHandler * dh = new DataHandler(path, pathGabor);

    cout<<"Loading files"<<endl;
    dh->load();
    cout<<"Loaded files:"<<dh->all_data.size()<<endl;
    dh->saveGaborImages(dh->all_data.at(53));
    dh->saveAreaImage(dh->all_data.at(3));
    dh->saveAreaImage(dh->all_data.at(53));
    //dh->saveHistImage(dh->all_data.at(53));//->histogram, "./../gabor/hist.jpg")

    int set = 70;  //ratio 1:n-1
    int cycles = 20;
    int k = 0;
    vector<double> results, res_xor, res_e, res_hist;
    vector <Image *> wrong, right;
    while(k < cycles){
        wrong.clear();
        right.clear();
        dh->extractSets(set);
        cout<<"Training files:"<<dh->training_set.size()<<endl;
        cout<<"Testing files:"<<dh->testing_set.size()<<endl;

        //break;
        int ok = 0, ok_h = 0, ok_e = 0, ok_xor = 0, ok_hist = 0;

        gh->fill(dh->training_set);
        //gh->trainAllVsAll();
        double percent_h, percent_xor, percent_e, percent_hist;
        int file_cnt = 1;
        int testing = dh->testing_set.size();

        for(auto t : dh->testing_set){
            //ok_xor += gh->fitXOR(t);
            //ok_h += gh->fitHD(t);
            //int l = gh->fitED(t);
            int l = gh->fitEDdist(t);
            if(l == 0){
                wrong.push_back(t);
            }
            else{
                right.push_back(t);
            }
            ok_e += l;
            cout<<"\nFILE: "<<file_cnt<<"/"<<testing<<endl;
            cout<<"\nCORRECT: "<<ok_e<<"/"<<testing<<endl;
            //cout<<"\nCORRECT: "<<ok_e<<"/"<<testing<<endl;
            /*
            percent_xor = (((double)ok_xor/(double)testing)*100);
            cout<<"PERCENTAGE (XOR): "<<percent_xor<<" %"<<endl;

            percent_h = (((double)ok_h/(double)testing)*100);
            cout<<"PERCENTAGE (HD): "<<percent_h<<" %"<<endl;
            */
            percent_e = (((double)ok_e/(double)testing)*100);
            cout<<"PERCENTAGE (ED): "<<percent_e<<" %"<<endl;
            file_cnt++;

            ok_hist += simpleHistogram(dh->training_set, t);
            cout<<"\nCORRECT: "<<ok_hist<<"/"<<testing<<endl;
            percent_hist = (((double)ok_hist/(double)testing)*100);
            cout<<"PERCENTAGE (HIST): "<<percent_hist<<" %"<<endl;
        }
        //results.push_back(percent_h);
        //res_xor.push_back(percent_xor);
        res_e.push_back(percent_e);
        res_hist.push_back(percent_hist);
        dh->clearSets();
        gh->clear();
        k++;
    }

    for(auto a: wrong){
        a->saveImage(pathWrong);
    }

    for(auto a: right){
        a->saveImage(pathRight);
    }

    for(auto a: dh->training_set){
        a->saveImage("./../train/");
    }

    cout<<"RESULTS: "<<endl;
    double all = 0, all_xor = 0, all_e = 0, all_hist=0;
    double std = 0;
    /*
    for(auto res: results){
        cout<<"HD "<<res<<" %"<<endl;
        all+=res;
    }
    double avg = all/(double)cycles;


    for(auto res: res_xor){
        cout<<"XOR "<<res<<" %"<<endl;
        all_xor+=res;
    }
    */

    for(auto res: res_hist){
        cout<<"HIST "<<res<<" %"<<endl;
        all_hist+=res;
    }
    double avg_hist = all_hist/(double)cycles;


    for(auto res: res_e){
        cout<<"ED "<<res<<" %"<<endl;
        all_e+=res;
    }
    double avg_e = all_e/(double)cycles;
    for(auto res: res_e){
        std += ((res - avg_e)*(res - avg_e));
    }
    std = std/(double)cycles;
    std = sqrt(std);

    cout<<endl<<"SETS: "<<set<<endl;
    cout<<"AVERAGE (ED): "<<avg_e<<endl;
    cout<<"STD ED: "<<std<<endl<<endl;
    cout<<"AVERAGE (HIST): "<<avg_hist<<endl;

    return 0;
}



