//
//  MultiLayerPerceptron.cpp
//  AA1
//
//  Created by Maurizio Idini on 05/01/15.
//  Copyright (c) 2015 Maurizio Idini. All rights reserved.
//

#include <cassert>
#include <iostream>
#include "DataSet.cpp"
#include "gnuplot_i.hpp"
#include <math.h>

#define MAX_ERR_PLOT 2000
#define MIN_ERR 0.01 
// #define DEBUG 

using namespace std;

struct Neuron {
    float x; //neuron input value
    float t; //target value
    float o; //output value
    float *w; //weight value
    float g; //local gradient
    float *dw; //deltaweight al passo precedente
    float *wBackup; //backup weights
    float wBias; //connessione dal neurone corrente al bias del layer precedente
    float dBias; //deltaweight al passo precedente del bias
    float biasBackup; //backup weight
};
struct Layer {
    int unitNumber;
    Neuron * neuron;
};

class MultiLayerPerceptron {
public:
    MultiLayerPerceptron(DataSet dataset, float eta, float alpha, float lambda, int maxEpochs, bool isRegression=true);
    void trainingCV();
    void training();
    void addExternalTestSet(DataSet TS);
    // void validate();
    void blind(char *filename);
    void saveWeightsIntoFile(char *filename);
    void restoreWeightFromFile(char *filename);
    void savePlotName(string filename);
    float getMEEValue();
    float getAccuracy();
private:
    //determina se il problema è di regressione o classificazione
    bool _isRegression;
    int kFold;
    int nLayer;
    Layer *layers;
    DataSet trainingSet;
    DataSet testSet;
    int _maxEpochs;
    float _eta;
    float _alpha;
    float _lambda;
    //variabili su cui memorizzare l'errore e l'accuratezza della rete
    float MEE;
    float accuracy;
    //vettori su cui memorizzare tutti gli errore e i valori di accuratezza della rete
    vector<float> MEEVec;
    vector<float> accVec;
    #ifdef DEBUG
    ofstream log;
    #endif

    void inizializeMlp();
    void inizializeWeigth();
    float randomWeigthValue(float low, float high);
    void setInput(vector<float> patternVector);
    void backupWeight();
    void restoreWeight();
    void setTarget(vector<float> patternVector);
    float actFun(float x);
    float derivatFun(float x);
    float computeError();
    void forward();
    void backward();
    void updateWeight();
    void train();
    void test();
    vector<float> getOutput();
};

/*
 * costruttore
 */
MultiLayerPerceptron::MultiLayerPerceptron(DataSet dataset, float eta, float alpha, float lambda, int maxEpochs, bool isRegression){
    #ifdef DEBUG
    log.open("log.txt");
    #endif
    trainingSet = dataset;
    nLayer = trainingSet.topology.size();
    layers = new Layer[nLayer];
    _eta = eta;
    _alpha = alpha;
    _lambda = lambda;
    _maxEpochs = maxEpochs;
    inizializeMlp();
    inizializeWeigth();
    _isRegression = isRegression;
    kFold = 5;
    if(_isRegression) 
        cout << "Task is Regression\n";
    else 
        cout << "Task is Classification\n";
}




/*
 * inizializeMlp
 * inizializza la rete in base al vettore topology
 */
void MultiLayerPerceptron::inizializeMlp() {
    for (int i=0; i < nLayer; i++){
        layers[i].unitNumber = trainingSet.topology[i];
        layers[i].neuron = new Neuron[layers[i].unitNumber];
        for (int j=0; j < layers[i].unitNumber; j++){
            layers[i].neuron[j].x = 1.0;
            layers[i].neuron[j].o = 0.0;
            layers[i].neuron[j].t = 0.0;
            layers[i].neuron[j].g = 0.0;
            if(i>0){
                layers[i].neuron[j].w = new float[layers[i-1].unitNumber];
                layers[i].neuron[j].dw = new float[layers[i-1].unitNumber];
                layers[i].neuron[j].wBackup = new float[layers[i-1].unitNumber];
                layers[i].neuron[j].wBias = 0.0;
                layers[i].neuron[j].dBias = 0.0;
                layers[i].neuron[j].biasBackup = 0.0;
            }       
        }
    }
}

//____________________WEIGHTS__________________________________
/*
 * randomWeigthValue
 * restituisce un valore compreso fra low e high
 */
float MultiLayerPerceptron::randomWeigthValue(float low, float high)
{
    srand(rand()+ time(0)); //necessario per una vera randomizzazione
    //return roundf( (((float) rand() / RAND_MAX + 1) * (high-low) + low )*100 )/100 ;
    return static_cast <float> (rand() / (RAND_MAX /(high-low))) + low;
}
/*
 * inizializeWeight
 * assegna ai pesi di ogni singolo neurone un valore random fra -0.5 e 0.5
 */
void MultiLayerPerceptron::inizializeWeigth(){
    #ifdef DEBUG
    log << "inizialize weights\n";
    #endif
    for (int i=1; i < nLayer; i++)
        for (int j=0; j < layers[i].unitNumber; j++) {
            layers[i].neuron[j].wBias = randomWeigthValue(-0.5, 0.5);
            #ifdef DEBUG
            log << "bias: " << layers[i].neuron[j].wBias << "\n";
            #endif
            for(int k=0; k < layers[i-1].unitNumber; k++){ 
                layers[i].neuron[j].dw[k]=0.0;
                layers[i].neuron[j].wBackup[k] = 0.0;
                layers[i].neuron[j].w[k] = randomWeigthValue(-0.5, 0.5); 
                #ifdef DEBUG
                log << " " << layers[i].neuron[j].w[k];        
                #endif   
            }
            #ifdef DEBUG
            log << "\n";
            #endif
        }
        #ifdef DEBUG
        log << "end inizialize weights ***********************************\n";
        #endif
}
/*
 * backupWeight
 * salva lo stato corrente dei pesi internamente al neurone
 */
void MultiLayerPerceptron::backupWeight(){
    for (int i=1; i < nLayer; i++)
        for (int j=0; j < layers[i].unitNumber; j++){
            layers[i].neuron[j].biasBackup = layers[i].neuron[j].wBias;
            for(int k=0; k<layers[i-1].unitNumber; k++)
                layers[i].neuron[j].wBackup[k] = layers[i].neuron[j].w[k];
        }
}
/*
 * restoreWeight
 * ripristina lo stato dei pesi
 */
void MultiLayerPerceptron::restoreWeight(){
    for (int i=1; i < nLayer; i++)
        for (int j=0; j < layers[i].unitNumber; j++) {
            layers[i].neuron[j].wBias = layers[i].neuron[j].biasBackup;
            for(int k=0; k<layers[i-1].unitNumber; k++) 
                layers[i].neuron[j].w[k] = layers[i].neuron[j].wBackup[k];
        }
}
/*
 * saveWeightsIntoFile
 * salva i pesi in un file
 */
void MultiLayerPerceptron::saveWeightsIntoFile(char *filename){ 
    ofstream f(filename);
    for (int i=1; i < nLayer; i++){
        for (int j=0; j < layers[i].unitNumber; j++){
            for(int k=0; k<=layers[i-1].unitNumber; k++)
                f << layers[i].neuron[j].w[k] << "\n";
            f << layers[i].neuron[j].wBias << "\n";
            }
    }
    f.close();
}
/*
 * restoreWeightFromFile
 * ripristina i pesi da un file
 */
void MultiLayerPerceptron::restoreWeightFromFile(char *filename){  
    ifstream file;
    file.open(filename);
    string line;
    string token;
    for (int i=1; i < nLayer; i++)
        for (int j=0; j < layers[i].unitNumber; j++){
            for(int k=0; k<=layers[i-1].unitNumber; k++){
                std::getline(file, line);
                std::istringstream iss(line);
                iss >> token;
                layers[i].neuron[j].w[k]=atof(token.c_str());
            }
            std::getline(file, line);
            std::istringstream iss(line);
            iss >> token;
            layers[i].neuron[j].wBias=atof(token.c_str());
        }
    file.close();
}


//____________________WEIGHTS__________________________________//

/*
 * setInput
 * imposta l'input layer con i valori contenuti nel vettore in input
 */
void MultiLayerPerceptron::setInput(vector<float> patternVector){
    //assing input value to input layer
    #ifdef DEBUG
    log << "Set input \n";
    #endif
    for(int j=0; j<layers[0].unitNumber; j++){
        layers[0].neuron[j].x=patternVector[j];    
        #ifdef DEBUG
        log << layers[0].neuron[j].x << " ";
        #endif
    }
    #ifdef DEBUG
    log << "\n";
    #endif
}
/*
 * setTarget
 * imposta l'output layer con i valori contenuti nel vettore in input
 */
void MultiLayerPerceptron::setTarget(vector<float> targetVector){
    //assing target value to output layer
    #ifdef DEBUG
    log << "set target \n";
    #endif
    for(int j=0; j<layers[nLayer-1].unitNumber; j++){
        layers[nLayer-1].neuron[j].t = targetVector[j];
        #ifdef DEBUG
        log << layers[nLayer-1].neuron[j].t << " ";
        #endif
    }
    #ifdef DEBUG
    log << "\n";
    #endif
}



/*
 * actFun
 * applica la funzione di attivazione sigmoidea al valore in input
 */
float MultiLayerPerceptron::actFun(float x){
    float output = 1.0 / (1.0 + exp(-x));//tanh(x);
    #ifdef DEBUG
    log << "apply Act Fun, input : " << x << " output : " << output << " \n"; 
    #endif
    return output; 
}

float MultiLayerPerceptron::derivatFun(float x){
    float output = x * (1.0 - x);//1.0 - x * x;
    #ifdef DEBUG
    log << "apply derivative, input : " << x << "output : " << output << " \n"; 
    #endif
    return output;
    
}
/*
 * forward
 * propaga il segnale in avanti
 */
void MultiLayerPerceptron::forward(){
    #ifdef DEBUG
    log << "propagate signal \n";
    #endif 
    for(int i=1; i<nLayer; i++){
        #ifdef DEBUG
        log << "layer " << i << "\n";
        #endif
        for(int j=0; j<layers[i].unitNumber; j++){
            #ifdef DEBUG
            log << "neuron " << j << "\n";
            #endif
                float sum = layers[i].neuron[j].wBias;
                #ifdef DEBUG
                log << "sum = " << sum << " + \n";
                #endif
                for (int k=0; k<layers[i-1].unitNumber; k++){
                    float x = layers[i-1].neuron[k].x;
                    float w = layers[i].neuron[j].w[k];
                    #ifdef DEBUG
                    log << "  " << x << " * " << w << "\n";
                    #endif
                    sum += x * w;
                }
                #ifdef DEBUG
                log << "   = " << sum << "\n";
                #endif
                if(_isRegression){
                    if(i!=nLayer-1) { 
                        layers[i].neuron[j].x = actFun(sum); 
                    }
                    else { 
                        layers[i].neuron[j].x = sum; 
                    }
                } else {
                    layers[i].neuron[j].x = actFun(sum);
                }
        }
    }
    #ifdef DEBUG
    log << "end Propagate signal ******************************************\n";
    #endif
}
/*
 * computeError
 * calcola l'errore che commette la rete e lo memorizza nella variabile in input
 */
float MultiLayerPerceptron::computeError(){
    //compute error and store it
    float err = 0.0;
    for (int j=0; j<layers[nLayer-1].unitNumber; j++){
        float x = layers[nLayer-1].neuron[j].x;
        float t = layers[nLayer-1].neuron[j].t;
        err += (x - t)*(x - t);
    } 
    return sqrt(err) ;
}
/*
 * backward
 * calcola l'errore e lo propaga all'indietro
 */
void MultiLayerPerceptron::backward(){
    #ifdef DEBUG
    log << "backward\n";
    #endif
    //compute output gradient
    #ifdef DEBUG
    log << "layer " << nLayer-1 << "\n";
    #endif
    for(int i=0; i<layers[nLayer-1].unitNumber; i++) {
        #ifdef DEBUG
        log << "neuron " << i << "\n";
        #endif
        float x = layers[nLayer-1].neuron[i].x;
        float t = layers[nLayer-1].neuron[i].t;
        #ifdef DEBUG
        log << "output: " << x << " target: " << t <<"\n";
        #endif
        if(_isRegression)
            layers[nLayer-1].neuron[i].g = (t - x) ;
        else layers[nLayer-1].neuron[i].g = (t - x) * derivatFun(x);
        #ifdef DEBUG
        log << "gradient: " << layers[nLayer-1].neuron[i].g << "\n";
        #endif
    }
    // compute hidden gradient
    for(int i=nLayer-2; i>0; i--) {
        #ifdef DEBUG
        log << "layer " << i << "\n";
        #endif
        for(int j=0; j<layers[i].unitNumber; j++) {
            #ifdef DEBUG
            log << "neuron " << i << "\n";
            #endif
            float x = layers[i].neuron[j].x;
            float sum = 0.0;
            #ifdef DEBUG
            log << "sum = ";
            #endif
            for (int k=0; k<layers[i+1].unitNumber; k++) {
                sum += layers[i+1].neuron[k].w[j] * layers[i+1].neuron[k].g;
                #ifdef DEBUG
                log << layers[i+1].neuron[k].w[j] << " * " << layers[i+1].neuron[k].g << " ";
                #endif
            }
            #ifdef DEBUG
            log << "  = " << sum << "\n";
            #endif
            layers[i].neuron[j].g = sum * derivatFun(x);
            #ifdef DEBUG
            log << "gradient: " << layers[i].neuron[j].g << "\n";
            #endif
        }
    }
    #ifdef DEBUG
    log << "end backward ******************************************\n";
    #endif
}

/*
 * updateWeight
 * aggiorna il valore dei pesi
 */
void MultiLayerPerceptron::updateWeight(){
    //w_ji(k+1) = w_ji(k) + DELTAw_ji(k) + ALPHA * DELTAw_ji(k-1)
    //DELTAw_ji = eta gradient_j output_i
    for(int i=1; i<nLayer; i++)
        for(int j=0; j<layers[i].unitNumber; j++){
            float g = layers[i].neuron[j].g;

            layers[i].neuron[j].wBias *= (1.0 - _lambda);
            layers[i].neuron[j].wBias += _eta * g + _alpha * layers[i].neuron[j].dBias;
            layers[i].neuron[j].dBias = _eta * g ;

            for (int k=0; k<layers[i-1].unitNumber; k++){
                layers[i].neuron[j].w[k] *= (1.0 - _lambda);
                layers[i].neuron[j].w[k] += 
                    _eta * layers[i-1].neuron[k].x * g + _alpha * layers[i].neuron[j].dw[k];
                layers[i].neuron[j].dw[k] = _eta * layers[i-1].neuron[k].x * g ;
            }
        }
}

void MultiLayerPerceptron::train(){
    for(unsigned i=0; i<trainingSet.patternNumber; ++i){   
        setInput(trainingSet.patternVector[i].patternElements);
        setTarget(trainingSet.patternVector[i].outputPattern);
        forward();
        backward();
        updateWeight();
    } //end for pattern
}

void MultiLayerPerceptron::test() {
    #ifdef DEBUG
    log << "************ TEST *********************\n";
    #endif
    int accuracyTemp = 0;
    float testError = 0.0;
    for(unsigned i=0; i<testSet.patternVector.size(); ++i){   
        setInput(testSet.patternVector[i].patternElements);
        setTarget(testSet.patternVector[i].outputPattern);
        forward();

        if(!_isRegression){
            for (int j=0; j<layers[nLayer-1].unitNumber; ++j){
                float t = layers[nLayer-1].neuron[j].t;
                float x = layers[nLayer-1].neuron[j].x;
                if(x >= 0.5 && t >= 0.5) accuracyTemp++;
                else if(x < 0.5 && t < 0.5) accuracyTemp++;
            }
        }
        testError += computeError();
    } //end for pattern
    MEE=testError/(float)testSet.patternNumber;
    //controllo MEETrain is nan
    if(MEE!=MEE) {
        cout << "NOT A NUMBER ERROR" << endl;
        MEE = MAX_ERR_PLOT;
    }
    if(!_isRegression)
        accuracy=accuracyTemp/(float)testSet.patternNumber;
    #ifdef DEBUG
    log << "************ END TEST *********************\n";
    #endif 
}


void MultiLayerPerceptron::trainingCV(){
    float MEEavg = 0.0;
    float accuracyAvg = 0.0;
    int epoch = 0;
    float currentError = 0.0;
    int k_min=0;
    int k_max = 0;
    int patternNum = trainingSet.patternNumber;
    int sizeFold = (trainingSet.patternNumber / kFold) + 1;
    DataSet TS;
    for(int i=0; i<kFold; i++){
        cout << "Fold " << i << endl;
        k_min = i*sizeFold;
        k_max = (i+1)*sizeFold;
        if(k_max > patternNum) k_max = patternNum;
        DataSet::splitDataSetFromMinToMax(trainingSet,testSet,k_min,k_max);

        while (epoch != _maxEpochs){
            train();
            test();
            if(epoch==0) 
                currentError = MEE;
            epoch++;
            if(MEE<currentError )
                currentError = MEE;
            if(MEE > 1.2 * currentError){
            }
            MEEVec.push_back(MEE);
            if(!_isRegression) accVec.push_back(accuracy);
            trainingSet.randomizeElements();
            // if (MEE < MIN_ERR) {
            //     break;
            // }
        }//end while
        cout << "  Stop at epoch no. " << epoch << endl;
        epoch=0;
        cout << "  Current Error is " << MEE << endl;
        MEEavg += MEE;
        if(!_isRegression)
            accuracyAvg += accuracy;
        DataSet::unionTR_TS(trainingSet,testSet,k_min);
    } //end for kFold+
    MEE = MEEavg / kFold;
    if(!_isRegression)
        accuracy = accuracyAvg / kFold;

    vector<float> finalMEEVec;
    vector<float> finalAccuracy;
    //calcolo media vettori
    for(unsigned i = 0; i < _maxEpochs; i++){
        float sum = 0.0;
        float sumA = 0.0;
        for(unsigned j = 0; j < kFold; j++){
            sum += MEEVec[i+j*_maxEpochs];
            if(!_isRegression)
                sumA += accVec[i+j*_maxEpochs];
        }
        finalMEEVec.push_back(sum/kFold);
        if(_isRegression) finalAccuracy.push_back(sumA/kFold);
    }
    MEEVec = finalMEEVec;

    #ifdef DEBUG
    log.close();
    #endif
}

void MultiLayerPerceptron::training(){
    int epoch = 0;
    float currentError = 0.0;
    if(testSet.patternNumber==0){
        cout << "No testSet. Split \n";
        DataSet::splitDataSet(trainingSet,testSet,25);
    }
    while(epoch != _maxEpochs){
        train();
        test();
        if(epoch==0) currentError = MEE;
        epoch++;
        if(MEE<currentError )
            currentError = MEE;
        if(MEE > 1.2 * currentError){
            // break;
        }
        MEEVec.push_back(MEE);
        if(!_isRegression) accVec.push_back(accuracy);
        trainingSet.randomizeElements();
        // if (MEE < MIN_ERR) {
        //     break;
        // }
    }//end while
    cout << "Stop at epoch no. " << epoch << endl;

    #ifdef DEBUG
    log.close();
    #endif
}

void MultiLayerPerceptron::addExternalTestSet(DataSet TS){
    testSet = TS;
}
// void MultiLayerPerceptron::validate() {

// }


/*
 * savePlotName
 * salva il plot risultante dai vettori di errore ed eventualmente dai vettori di accuracy
 * se Gnuplot non è installato nella macchina in esecuzione, ignora e va avanti
 */
void MultiLayerPerceptron::savePlotName(string filename){
    if(Gnuplot::get_program_path()){
        Gnuplot gp("lines");
        gp.savetopng(filename+"_MEE");
        gp.cmd("set xlabel \"Epochs\"");
        gp.cmd("set ylabel \"MEE\"");
        if(MEEVec.size()==0) MEEVec.push_back(0);
        gp.plot_x(MEEVec,"MEE");
        if(!_isRegression) {
            gp.reset_plot();
            gp.cmd("set ylabel \"Accuracy\"");
            gp.savetopng(filename+"_accuracy");
            gp.plot_x(accVec,"accuracy");
        }
    } else {
        cerr << "Gnuplot is not installed in your machine. Plot not saved." << endl;
        cerr << "Don't worry!! Computation continues!" << endl;
    }
}
/*
 * getMEETestValue
 * restituisce il valore finale dell'error test
 */
float MultiLayerPerceptron::getMEEValue() {
    return MEE;
}
/*
 * getAccuracy
 * restituisce il valore finale dell'accuracy
 */
float MultiLayerPerceptron::getAccuracy(){
    return accuracy;
}
/*
 * getOutput
 * restituisce il vettore dell'output layer con i risultati della rete
 */
vector<float> MultiLayerPerceptron::getOutput(){
    vector<float> output;
    for(int j=0; j<layers[nLayer-1].unitNumber; ++j){
        output.push_back(layers[nLayer-1].neuron[j].x);
    }
    return output;
}


/*
 * runBlindTest
 * esegue solo la propagazione del segnale con i dati del file blind test
 * e li scrive in un nuovo file
 */
void MultiLayerPerceptron::blind(char* filename){
    ofstream blind(filename);
    blind << "# Maurizio Idini \n"
          // << "# nome team \n"
          << "# LOC-SM - AA1 2014 CUP v1 \n"
          << "# 06/02/2015 \n" ;
    for(unsigned i=0; i<trainingSet.patternVector.size(); ++i){
    setInput(trainingSet.patternVector[i].patternElements);
    forward();
    vector<float> output = getOutput();
    blind << trainingSet.patternVector[i].instanceNumber;
    for(unsigned i=0; i<output.size(); ++i)
       blind << "," << output[i];
    blind << "\n";
    }
    blind.close();
}

// int main(int argc, const char * argv[]) {
//     #ifdef DEBUG
//     printf("Usage: \ndataset #input #hidden #output eta alpha lambda #epochs plotname \n");
//     #endif
// 	DataSet trainDS;
// 	 // vector<int> topologyMonk = {17,3,1};
//      // int epochs=390;
//     vector<int> topology = {17,4,1};
// 	string monkName="monk3";
//     int epochs=190;
// 	trainDS.createDataSetFromMonk("../dataset/monks-3.train.txt",topology);
//     DataSet test;
//     test.createDataSetFromMonk("../dataset/monks-3.test.txt",topology);
//     // trainDS.createTrSetLogic(argv[1], topology,false);
// 	MultiLayerPerceptron mlp(trainDS,atof(argv[1]),atof(argv[2]),atof(argv[3]),epochs,false);
//     mlp.addExternalTestSet(test);
// 	mlp.training();
//     cout << "Accuracy is " << mlp.getAccuracy() << endl;
//     cout << "Error is " << mlp.getMEEValue() << endl;
//     mlp.savePlotName(monkName);
// 	}
