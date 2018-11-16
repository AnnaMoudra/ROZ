#include <cstdlib>
#include <cstring>
#include <string>
#include <dirent.h>
#include <iostream>


using namespace std;

int main()
{
    DIR * dirp;
    string path = "./../iris_all";
    dirp = opendir(path);
    while (readdir(dirp) != NULL) {
        cout << readdir(dirp)->d_name << endl;


    }
    (void)closedir(dirp);
    return 0;
}