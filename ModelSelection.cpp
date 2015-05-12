//
//  ModelSelection.cpp
//  AA1
//
//  Created by Maurizio Idini
//  Copyright (c) 2015 Maurizio Idini. All rights reserved.
//

#include "MultiLayerPerceptron.cpp"

class ModelSelection {
public:
    ModelSelection(DataSet training, DataSet validation, bool regressionTask = true);
    void setHyperParametersFromFile(char * fileName);
    // void setEtaWindow(double eta_min, double eta_max);
    // void setAlphaWindow(double alpha_min, double alpha_max);
    // void setLambdaWindow(double lambda_min, double lambda_max);
    // void setHiddenUnitWindow(int unitMin, int unitMax);
    void startSelection(char *fileResult);
    void startAssessment(char *fileSelection, char *fileAssess);
    double getEta();
    double getAlpha();
    double getLambda();
    int getHiddenUnitNumber();
    int getMaxEpoch();
    double getMEE();
private:
    DataSet trainSet; //training
    DataSet validationSet; //validation
    double alphaMin;
    double alphaMax;
    double etaMin;
    double etaMax;
    double lambdaMin;
    double lambdaMax;
    int hUnitMin;
    int hUnitMax;
    double etaFinal;
    double alphaFinal;
    double lambdaFinal;
    int hUnitFinal;
    double minMEE;
    bool isRegression;
    int maxEpoch;
};
/*
 * costruttore
 * crea i vai dataset come da modello
 */
ModelSelection::ModelSelection(DataSet training, DataSet validation, bool regressionTask){
    trainSet = training;
    validationSet = validation;
    isRegression=regressionTask;
}
/*
 * setEtaWindow
 * imposta il range di valori
 */
// void ModelSelection::setEtaWindow(double eta_min, double eta_max){
//     etaMin=eta_min;
//     etaMax=eta_max;
// }
/*
 * setAlphaWindow
 * imposta il range di valori
 */
// void ModelSelection::setAlphaWindow(double alpha_min, double alpha_max){
//     alphaMin=alpha_min;
//     alphaMax=alpha_max;
// }
/*
 * setLambdaWindow
 * imposta il range di valori
 */
// void ModelSelection::setLambdaWindow(double lambda_min, double lambda_max){
//     lambdaMin=lambda_min;
//     lambdaMax=lambda_max;
// }
/*
 * setHiddenUnitWindow
 * imposta il range di valori
 */
// void ModelSelection::setHiddenUnitWindow(int unitMin, int unitMax){
//     hUnitMin=unitMin;
//     hUnitMax=unitMax;
// }


double ModelSelection::getEta() { return etaFinal; }
double ModelSelection::getAlpha() { return alphaFinal; }
double ModelSelection::getLambda() { return lambdaFinal; }
int ModelSelection::getHiddenUnitNumber(){ return hUnitFinal; }
double ModelSelection::getMEE() { return minMEE; }
int ModelSelection::getMaxEpoch() { return maxEpoch; }

void ModelSelection::setHyperParametersFromFile(char * fileName){
    //structure of file is:
    //eta_min eta_max alpha_min alpha_max lambda_min lambda_max hu_min hu_max maxEpoch
    ifstream file(fileName);
    string line;
    string token;
    if(file.good()){
        std::getline(file, line);
        std::istringstream iss(line);
        iss >> token;
        etaMin=atof(token.c_str());
        iss >> token;
        etaMax=atof(token.c_str());
        alphaMin=atof(token.c_str());
        iss >> token;
        alphaMax=atof(token.c_str());
        lambdaMin=atof(token.c_str());
        iss >> token;
        lambdaMax=atof(token.c_str());
        hUnitMin=atof(token.c_str());
        iss >> token;
        hUnitMax=atof(token.c_str()); 
        iss >> token;
        maxEpoch = atof(token.c_str());
    }
    file.close();
}



/*
 * startSelection
 * si occupa di fare Model Selection in base al range valori settati precedentemente
 */
void ModelSelection::startSelection(char *fileResult){
    bool firstIteration = true;
    long iterations=0;
    double currentError=0.0;
    ofstream log("modelSelection_log.txt");
    ofstream f(fileResult);
    for(int u=hUnitMin; u<=hUnitMax; u++){
        trainSet.setHiddenUnit(u);
        for(double l=lambdaMin; l<=lambdaMax; l+=lambdaMin) {
            for(double a=alphaMin; a<=alphaMax; a+=alphaMin) {
                for(double e=etaMin; e<=etaMax; e+=etaMin) {
                    cout << "iteration no: " << iterations << endl;
                    MultiLayerPerceptron mlp(trainSet,e,a,l,maxEpoch,isRegression);
                    mlp.trainingCV();
                    currentError = mlp.getMEEValue();
                    if(firstIteration) {
                        minMEE = currentError;
                        firstIteration=false;
                    }
                    if(currentError < minMEE) {
                        etaFinal=e; alphaFinal=a; lambdaFinal=l; hUnitFinal=u;
                        mlp.training();
                        minMEE = mlp.getMEEValue();
                        string x = to_string(iterations);
                        mlp.savePlotName(x.c_str());
                        log << "Iteration no: "<< iterations <<" Eta: " << etaFinal << " Alpha: " << alphaFinal
                          << " Lambda: " << lambdaFinal << " HiddenUnits: " << hUnitFinal
                          << " MEE: " << minMEE << "\n";
                        cout << "Iteration no: "<< iterations;
                        cout << " Eta "    << etaFinal << " Alpha " << alphaFinal
                        << " Lambda " << lambdaFinal << " HiddenUnits " << hUnitFinal
                        << " MEE " << currentError << "\n";
                    }
                    iterations++;
                }
            }
        }
    }
    f << etaFinal << " " << alphaFinal << " " << lambdaFinal << " " << hUnitFinal << "\n";
    f.close();
    log << "Total number of iterations is " << iterations << "\n";
    log.close();
}

/*
 * startAssessment
 * si occupa di fare Model Assessment in base ai risultati della Model Selection
 * richiede la partecipazione dell'utente nella scelta del modello finale
 */
void ModelSelection::startAssessment(char *fileSelection,char *fileAssess){
    double eta,alpha,lambda;
    int hiddenUnits;
    double etaFin,alphaFin,lambdaFin;
    int hiddenFin;
    double currentError;
    int iter=0;
    ifstream file(fileSelection);
    ofstream log("model_assessment_log.txt");
    ofstream out(fileAssess);
    string line;
    string token;
    if(file.good()){
        std::getline(file, line);
        std::istringstream iss(line);
        iss >> token;
        eta=atof(token.c_str());
        iss >> token;
        alpha=atof(token.c_str());
        iss >> token;
        lambda=atof(token.c_str());
        iss >> token;
        hiddenUnits=atoi(token.c_str());
        validationSet.setHiddenUnit(hiddenUnits);
        MultiLayerPerceptron mlp(validationSet,eta,alpha,lambda,maxEpoch,isRegression);
        mlp.training();
        currentError=mlp.getMEEValue();

        string x = to_string(iter)+"_MA";
        mlp.savePlotName(x.c_str());

    }
    file.close();
    log << "Minimum error in validation set is \n";
    log << etaFin << " " << alphaFin << " " << lambdaFin << " " << hiddenFin << "\n";
    log << minMEE << "\n";
    log.close();
    out << to_string(eta)+" "+to_string(alpha)+" "+to_string(lambda)+" "+to_string(hiddenUnits) << "\n";
    out.close();
}
