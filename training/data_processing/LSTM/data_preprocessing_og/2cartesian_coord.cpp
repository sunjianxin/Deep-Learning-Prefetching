//for any position of any size of volume
//g++ -std=c++11 step_4_LRU.cpp 
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <time.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <climits>
#include <map>
#include <unordered_set>
#include <chrono> 
#define PI 3.14159265

using namespace std;
using namespace std::chrono;


int main(int argc, char **argv){
    
    int test_num = 3400;
    
    ifstream infile;
    char fileName[50];
    char outName[50];

    for(int i = 1; i <= 5; i++) {
        sprintf (fileName, "test_%d_training_xyz.dat", i);
        sprintf (outName, "test_%d_training_xyz.txt", i);

        infile.open(fileName);
        assert(infile);
        float *dis = NULL;  //x
        dis = new float[test_num];
        float *theta = NULL; //y
        theta = new float[test_num];
        float *phi = NULL; //z
        phi = new float[test_num];
        
        infile.read(reinterpret_cast<char *>(dis), sizeof(float)*test_num);
        infile.read(reinterpret_cast<char *>(theta), sizeof(float)*test_num);
        infile.read(reinterpret_cast<char *>(phi), sizeof(float)*test_num);
        infile.close();

        ofstream outf;
        outf.open(outName, std::ofstream::out | std::ofstream::app);
        outf<<"x_0,y_0,z_0,x,y,z\n";

        for (int i=1; i<test_num; i++){

            outf<< std::to_string(dis[i-1])+","+std::to_string(theta[i-1])+","+std::to_string(phi[i-1])+","
            + std::to_string(dis[i])+","+std::to_string(theta[i])+","+std::to_string(phi[i])+"\n";
        }

        outf.close();
    }

 
   
    return 0;
}
