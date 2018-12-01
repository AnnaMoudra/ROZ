//50 by 50
//n-1 by 1
// 30-70
// in build: cmake --build . --config Release
// in SRC: .\build\Release\miroz.exe


#include "classes/DataHandler.cpp"
#include <opencv2/opencv.hpp>

//testing 5
using namespace std;

//return 0/1
// 1 vs SH dummmy 56%
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
    string path = "./../test_5";
    //string path = "./../iris_all";

    GaborHandler * gh = new GaborHandler();
    DataHandler * dh = new DataHandler(path);


    dh->load();
    cout<<"Loaded files:"<<dh->all_data.size()<<endl;
    dh->extractSets(1); //ratio 1:n-1
    cout<<"Training files:"<<dh->training_set.size()<<endl;
    cout<<"Testing files:"<<dh->testing_set.size()<<endl;

    Image * test;
    int ok = 0;

    gh->fill(dh->training_set);
    gh->trainAllVsAll();



    int file_cnt = 1;
    int testing = dh->testing_set.size();
    for(auto t : dh->testing_set){
        //fitting
        ok += gh->fit(t);
        cout<<"\nFILE: "<<file_cnt<<"/"<<testing<<endl;
        cout<<"\nCORRECT: "<<ok<<"/"<<testing<<endl;
        double percent = (((double)ok/(double)testing)*100);
        cout<<"PERCENTAGE: "<<percent<<" %"<<endl;
        file_cnt++;
    }

    //training_set.at(1)->drawHistChange();
    //training_set.at(4)->drawHistChange();
    //training_set.at(1)->EF_GABOR();
    //training_set.at(4)->showGabor();

    /*
    for(int l = 1; l < files; l++){
        test = training_set.at(l);
        vector <Image *> new_set (training_set);
        new_set.erase(new_set.begin()+l);

        ok += simpleHistogram(new_set, test);
        cout<<"\nCORRECT: "<<ok<<"/"<<files<<endl;
        double percent = (((double)ok/(double)files)*100);
        cout<<"PERCENTAGE: "<<percent<<" %"<<endl;
    }

    cout<<"Histogram finished"<<endl;
    cout<<"\nCORRECT: "<<ok<<"/"<<files<<endl;
    double percent = (((double)ok/(double)files)*100);
    cout<<"PERCENTAGE: "<<percent<<" %"<<endl;




    ok = 0;
    int test_no = 12;
    for(int l = 1; l < files; l++){
        test = training_set.at(l);
        vector <Image *> new_set (training_set);

        new_set.erase(new_set.begin()+l);
        cout<<"Start training on gabor feature vectors"<<endl;
        gh->clear();
        gh->fill(new_set);
        gh->trainAllVsAll();
        ok += gh->fit(test);
        cout<<"\nFILE: "<<l<<"/"<<files<<endl;
        cout<<"\nCORRECT: "<<ok<<"/"<<files<<endl;
        double percent = (((double)ok/(double)files)*100);
        cout<<"PERCENTAGE: "<<percent<<" %"<<endl;
    }



    */


    return 0;
}



