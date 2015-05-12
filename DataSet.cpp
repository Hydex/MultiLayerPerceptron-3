//  DataSet.cpp
//  AA1
//
//  Created by Maurizio Idini
//  Copyright (c) 2015 Maurizio Idini. All rights reserved.
//

#include <stdio.h>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstring>
#include <sstream>

using namespace std;

struct pattern {
    int instanceNumber;
    vector<float> patternElements;
    vector<float> outputPattern;
};

class DataSet {
public:
    vector<pattern> patternVector;
    int patternNumber=0;
    vector<int> topology;
    DataSet() {}
    void createDataSetFromCSV(char *filename, vector<int> my_topology);
    vector<float> retokenize(float token, int parameter);
    void createDataSetFromMonk(const char *filename, vector<int> my_topology);
    void createTrSetLogic(const char *filename, vector<int> my_topology, bool isTestSet);//*error definition
    void createTrSetNot(const char *filename, vector<int> my_topology, bool isTestSet);//*error definition
    
    static void splitDataSet(DataSet &TR, DataSet &TS, int perc_of_new_DataSet);
    static void splitDataSetFromMinToMax(DataSet &TR, DataSet &TS, int min, int max);
    static void unionTR_TS(DataSet &TR, DataSet &TS, int position = -1);
    void randomizeElements();
    void setHiddenUnit(int numberUnits);

private:

};

/*
 * setHiddenUnit
 * imposta il numero di hidden units nel vettore topology
 */
void DataSet::setHiddenUnit(int numberUnits){
    topology[1]=numberUnits;
}
/*
 * createDataSetFromFile
 * crea un dataset parsando un file CVS e imposta la topologia
 */
void DataSet::createDataSetFromCSV(char *filename, vector<int> my_topology){
    cout << "start parsing" << endl;
    topology=my_topology;
    string buffer;
    patternNumber=0;
    ifstream file(filename);
    if(!file) cerr << "Error: failed to open file "<< filename << endl;
    while (file.good()) {
        getline ( file, buffer, '\n' );
        if (!buffer.empty() && buffer.substr(0,1)!="#") { //salta riga vuota e commenti
            string delimiter = ",";
            size_t numberToken=0;
            std::string token;
            pattern p;
            size_t pos = 0;
            do{
                pos = buffer.find(delimiter);
                token = buffer.substr(0, pos);
                if(numberToken==0)
                    p.instanceNumber = atoi(token.c_str());
                else {
                    if (numberToken<=topology[0])
                        p.patternElements.push_back(atof(token.c_str()));
                    else p.outputPattern.push_back(atof(token.c_str()));
                }
                buffer.erase(0, pos + delimiter.length());
                numberToken++;
            }
            while (pos != std::string::npos);

            // cout << "pattern : ";
            // for (int i=0; i< p.patternElements.size(); i++)
            //     cout << " " << p.patternElements[i];
            // cout << " output : ";
            // for (int i=0; i< p.outputPattern.size(); i++)
            //     cout << " " << p.outputPattern[i];
            // cout << endl;
            patternVector.push_back(p);
            patternNumber++;
        }
    }
}

/*
 * splitDataSet ****************DEPRECATED*************************
 * suddivide il dataset sul quale verra' richiamato in base alla percentuale
 * in input. Restituisce un nuovo dataset e modifica il dataset corrente.
 */
void DataSet::splitDataSet (DataSet &TR, DataSet &TS, int perc_of_new_DataSet) { //side_effect su v1
    long numRows_v1 = TR.patternVector.size() * (100-perc_of_new_DataSet) / 100;
    numRows_v1 = TR.patternVector.size() - numRows_v1;
    // vector<pattern> v(make_move_iterator(TR.patternVector.begin()+numRows_v1),make_move_iterator(TR.patternVector.end()));
    vector<pattern> v(make_move_iterator(TR.patternVector.begin()+numRows_v1),make_move_iterator(TR.patternVector.begin()+(numRows_v1*2)));
    TS.patternVector = v;
    TR.patternVector.erase(TR.patternVector.begin()+numRows_v1,TR.patternVector.begin()+(numRows_v1*2));
    TR.patternNumber = TR.patternVector.size();
    TS.patternNumber = TS.patternVector.size();
    TS.topology=TR.topology;
}
// ********NEW*************
void DataSet::splitDataSetFromMinToMax(DataSet &TR, DataSet &TS, int min, int max){
    vector<pattern> v(make_move_iterator(TR.patternVector.begin()+min),make_move_iterator(TR.patternVector.begin()+max));
    TS.patternVector = v;
    TR.patternVector.erase(TR.patternVector.begin()+min,TR.patternVector.begin()+max);
    TR.patternNumber = TR.patternVector.size();
    TS.patternNumber = TS.patternVector.size();
    TS.topology=TR.topology;
}
// ********NEW*************
void DataSet::unionTR_TS(DataSet &TR, DataSet &TS, int position){
    if(position == -1) position = TR.patternNumber;
    vector<pattern>::iterator it = TR.patternVector.begin();
    TR.patternVector.insert(it+position, TS.patternVector.begin(), TS.patternVector.end());
    TR.patternNumber = TR.patternVector.size();
}

/*
 * randomizeElements
 * randomizza gli elementi del patternVector
 * NON E' POSSIBILE USARE RANDOM_SHUFFLE DI LIBC++11 PER PROBLEMI DI COMPATIBILITA'
 */
void DataSet::randomizeElements(){
    srand(rand()+ time(0)); //necessario per una vera randomizzazione
    for (unsigned i=0; i<patternVector.size(); ++i)
        swap(patternVector[i],patternVector[rand()*3%patternVector.size()]);
}

/*
 * createTrSetLogic
 * crea un dataset specifico per un problema logico a partire da un file testuale
 * dove ogni elemento è separato da uno spazio.
 * Esempio formato testo:
 * 1 1 1
 * 1 0 0
 * 0 1 0
 * 0 0 0
 */
void DataSet::createTrSetLogic(const char *filename, vector<int> my_topology, bool isTestSet){
    string token;
    ifstream file;
    file.open(filename);
    string line;
    string output;
    topology= my_topology;
    int countPattern=0;
    patternNumber=0;
    while(std::getline(file, line)) {
        pattern p;
        p.instanceNumber=countPattern++;
        std::istringstream iss(line);
        iss >> token;
        if(token=="0") p.patternElements.push_back(0.1);
        else p.patternElements.push_back(0.9);
        iss >> token;
        if(token=="0") p.patternElements.push_back(0.1);
        else p.patternElements.push_back(0.9);
        if(isTestSet==false){
            iss >> token;
            if(token=="0") p.outputPattern.push_back(0.1);
            else p.outputPattern.push_back(0.9);
        }
        // if(isTestSet==false){
        //     if(token=="0") p.patternElements.push_back(0.1);
        //     else p.patternElements.push_back(0.9);
        // }
        patternVector.push_back(p);
        patternNumber++;
    }
}

void DataSet::createTrSetNot(const char *filename, vector<int> my_topology, bool isTestSet){
    string token;
    ifstream file;
    file.open(filename);
    string line;
    string output;
    topology= my_topology;
    int countPattern=0;
    patternNumber=0;
    while(std::getline(file, line)) {
        pattern p;
        p.instanceNumber=countPattern++;
        std::istringstream iss(line);
        iss >> token;
        if(token=="0") p.patternElements.push_back(0.1);
        else p.patternElements.push_back(0.9);
        iss >> token;
        // if(token=="0") p.patternElements.push_back(0.1);
        // else p.patternElements.push_back(0.9);
        if(isTestSet==false){
            iss >> token;
            if(token=="0") p.outputPattern.push_back(0.1);
            else p.outputPattern.push_back(0.9);
        }
        // if(isTestSet==false){
        //     if(token=="0") p.patternElements.push_back(0.1);
        //     else p.patternElements.push_back(0.9);
        // }
        patternVector.push_back(p);
        patternNumber++;
    }
}
/*
 * retokenize
 * in base al token in input e il numero parametro, restituisce la codifica
 * del token e la inserisce nel vettore pattern
 */
vector<float> DataSet::retokenize(float token, int parameter){
    int length=0;
    switch(parameter){
    case 1 : length = 3; break;
    case 2 : length = 3; break;
    case 3 : length = 2; break;
    case 4 : length = 3; break;
    case 5 : length = 4; break;
    case 6 : length = 2; break;
    case 7 : length = 1; break;
    default : length = 0; break;
    }
    vector<float> aux;
    for(int i=length; i>0; --i){
            if(i==token) aux.push_back(0.9);
            else aux.push_back(0.1);
    }
    return aux;
}
/*
 * createDataSetFromMonk
 * crea un dataset specifico per MONK a partire da un file testuale
 * dove ogni elemento è separato da uno spazio
 */
void DataSet::createDataSetFromMonk(const char *filename, vector<int> my_topology){
    string token;
    ifstream file;
    file.open(filename);
    string line;
    string output;
    int countPattern=0;
    patternNumber=0;
    int cntParameter=1;
    topology=my_topology;

    // ofstream blind("monk1.txt");

    while(std::getline(file, line))
    {
        pattern p;
        p.instanceNumber=countPattern++;
        std::istringstream iss(line);
        iss >> output;

        while(iss >> token){
            if(token.find_first_not_of("01234")){
                vector<float> aux = DataSet::retokenize(atof(token.c_str()), cntParameter++);
                p.patternElements.insert(p.patternElements.end(),aux.begin(),aux.end());
                
            }
        }
        vector<float> aux = DataSet::retokenize(atof(output.c_str()), 7);
        p.outputPattern.insert(p.outputPattern.end(),aux.begin(),aux.end());
        for(int i=0; i<p.patternElements.size();i++){
            // blind << p.patternElements[i] << ",";
        }
        for(int i=0; i<p.outputPattern.size(); i++){
            // blind << p.outputPattern[i] << "\n";
        }
        patternVector.push_back(p);
        cntParameter=1;
        patternNumber++;
    }
    // blind.close();
}
