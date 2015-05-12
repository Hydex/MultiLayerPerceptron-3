//
//  NN-Training.cpp
//  AA1
//
//  Created by Maurizio Idini
//  Copyright (c) 2015 Maurizio Idini. All rights reserved.
//

#include <iostream>
#include <sys/time.h>
#include <ctime>
#include "NeuralNet.cpp"


using namespace std;


int main(int argc, char * argv[]) {
    struct timeval tm;
    struct timeval tm1;
    std::time_t start = std::time(nullptr);
    std::cout << "Time start " << std::asctime(std::localtime(&start)) << endl;
    gettimeofday( &tm, NULL );


//////Monk//////////////////////////////////////////
   // vector<int> topologyMonk = {17,3,1};
   // NeuralNet nn1("../dataset/monks-1.train.txt","./dataset/monks-1.test.txt",topologyMonk,"monk");
   // nn1.testMonk(1);
   // NeuralNet nn2("../dataset/monks-2.train.txt","./dataset/monks-2.test.txt",topologyMonk,"monk");
   // nn2.testMonk(2);
   // NeuralNet nn3("../dataset/monks-3.train.txt","./dataset/monks-3.test.txt",topologyMonk,"monk");
   // nn3.testMonk(3);
//////End Monk///////////////////////////////////////

//////XOR////////////////////////////////////////////
   // vector<int> topologyXor = {2,2,1};
   // NeuralNet nn4("xor.train.txt","xor.train.txt",topologyXor,"logic");
   // nn4.testXor();
//////End XOR////////////////////////////////////////

//////////Exploration Phase/////////////////////////////
    DataSet trainDS;
    vector<int> topology = {10,atoi(argv[5]),2};
    char * name=argv[6];
    trainDS.createDataSetFromCSV("./dataset/LOC-SM-TR.csv",topology);
    MultiLayerPerceptron mlp(trainDS,atof(argv[1]),atof(argv[2]),atof(argv[3]),atoi(argv[4]),true);
    mlp.trainingCV();
    cout << "Error is " << mlp.getMEEValue() << endl;
    mlp.savePlotName(name);
////////////////////////////////////////////////////////

/////////DataSet AA1////////////////////////////////////
    // vector<int> topologyAA1 = {10,10,2};
    // NeuralNet nnAA1("../dataset/LOC-SM-TR.csv","./dataset/LOC-SM-TS.csv",topologyAA1);
    // nnAA1.parameterSelection();
    // nnAA1.training();
    // nnAA1.blindTest("Idini_LOC-SM-TS.csv");
/////////End DataSet AA1////////////////////////////////

    gettimeofday(&tm1,NULL);
    cout << "time elapsed "<< floor((((double)tm1.tv_sec + (double)tm1.tv_usec / 1000000.0) - ((double)tm.tv_sec + (double)tm.tv_usec / 1000000.0))/60.0)
         << " minutes and " << floor(fmod(((double)tm1.tv_sec + (double)tm1.tv_usec / 1000000.0) - ((double)tm.tv_sec + (double)tm.tv_usec / 1000000.0),60.0))
         << " seconds." <<endl;
    std::time_t end = std::time(nullptr);
    std::cout << "Time end " << std::asctime(std::localtime(&end)) << endl;
    return 0;
}
