#include "ModelSelection.cpp"

class NeuralNet {
public:
    NeuralNet(char* trFile, char *blFile, vector<int> topology, char* problem="", bool regressionTask = true);
    void parameterSelection();
    void training();
    void blindTest(char *filename);
    void testMonk(int monk_dataset_number);
    void testXor();
private:
    void getParametersFromMA(char *filename);
    double eta;
    double alpha;
    double lambda;
    int hiddenUnits;
    DataSet trainDS;
    DataSet validDS;
    DataSet blindDS;
    bool isRegression;
    int maxEpoch;
};

/*
 * costruttore
 * richiama i giusti metodi per la creazione del dataset in base al valore booleano isCSV
 * default:bool isCsv=true
 */
NeuralNet::NeuralNet(char* trFile, char *blFile, vector<int> topology, char* problem, bool regressionTask){
    if(problem=="monk"){
        trainDS.createDataSetFromMonk(trFile,topology);
        validDS.createDataSetFromMonk(blFile,topology);
        isRegression=false;
    } else if(problem=="logic") {
        trainDS.createTrSetLogic(trFile, topology, false);
        DataSet::splitDataSet(trainDS,validDS,25);
        blindDS.createTrSetLogic(blFile, topology, true);
        isRegression=false;
    }
    else {
        trainDS.createDataSetFromCSV(trFile,topology);
        blindDS.createDataSetFromCSV(blFile,topology);
        isRegression=regressionTask;
    }
}

/*
 * modelSelection
 * passa il range di valori all'oggetto della classe ModelSelection
 * ed esegue i metodi startSelection e startAssessment
 */
void NeuralNet::parameterSelection() {
    cout << "Start Model Selection." << endl;
    ModelSelection modelSelect(trainDS,validDS,isRegression);
    modelSelect.setHyperParametersFromFile("grid_search.txt");
    maxEpoch = modelSelect.getMaxEpoch();
    modelSelect.startSelection("modelSelection_result.txt");
    cout << "Start Model Assessment." << endl;
    modelSelect.startAssessment("modelSelection_result.txt","modelAssessment_result.txt");

}
/*
 * training
 * prende i parametri finali ottenuti dalla model assessment ed
 * richiama il training della rete
 */
void NeuralNet::training() {
    getParametersFromMA("modelAssessment_result.txt");
    DataSet::unionTR_TS(trainDS,validDS);
    trainDS.setHiddenUnit(hiddenUnits);
    MultiLayerPerceptron mlp(trainDS,eta,alpha,lambda,maxEpoch,isRegression);
    mlp.training();
    mlp.savePlotName("training-result");
    cout << "MEE is "<< mlp.getMEEValue() << endl;
    mlp.saveWeightsIntoFile("weightBackup.txt");
}
/*
 * getParametersFromMA
 * si occupa di parsare i parametri finali dal file ottenuto dalla model assessment
 */
void NeuralNet::getParametersFromMA(char *filename){
    ifstream file;
    file.open(filename);
    string line;
    string token;
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
    file.close();
}

/*
 * blindTest
 * prende i parametri finali ottenuti dalla model assessment,
 * il backup dei pesi finali ed
 * esegue richiama il metodo runBlindTest
 */
void NeuralNet::blindTest(char* filename){
    getParametersFromMA("modelAssessment_result.txt");
    blindDS.setHiddenUnit(hiddenUnits);
    MultiLayerPerceptron mlp(blindDS,eta,alpha,lambda,maxEpoch,isRegression);
    mlp.restoreWeightFromFile("weightBackup.txt");
    mlp.blind(filename);
}



/*
 * testMonk
 * effettua i test sui dataset MONK
 */
void NeuralNet::testMonk(int monk_dataset_number){
    float eta,alpha,lambda;
    /* monk 1 hu 3 ep 390
     * monk 2 hu 2 ep 90
     * monk 3 hu 4 ep 190 (105 with weight decay)
     */
    int epochs=0;
    string monkName;
    switch(monk_dataset_number){
    case 1:
        hiddenUnits=3;
        trainDS.setHiddenUnit(hiddenUnits);
        validDS.setHiddenUnit(hiddenUnits);
        monkName="Monk-1";
        epochs=390;
        eta=0.5;
        alpha=0.9;
        lambda=0.00001;
        break;
    case 2:
        hiddenUnits=2;
        trainDS.setHiddenUnit(hiddenUnits);
        validDS.setHiddenUnit(hiddenUnits);
        monkName="Monk-2";
        epochs=90;        
        eta=0.25;
        alpha=0.9;
        lambda=0.00001;
        break;
    case 3:
        hiddenUnits=4;
        trainDS.setHiddenUnit(hiddenUnits);
        validDS.setHiddenUnit(hiddenUnits);
        monkName="Monk-3";
        epochs=190;        
        eta=0.1;
        alpha=0.5;
        lambda=0.00001;
        break;
    default :
        hiddenUnits=3;
        trainDS.setHiddenUnit(hiddenUnits);
        validDS.setHiddenUnit(hiddenUnits);
        monkName="Monk-1";
        epochs=390;
        eta=0.5;
        alpha=0.9;
        lambda=0.00001;
        break;
    }

    cout << monkName << endl;
    MultiLayerPerceptron mlp(trainDS,eta,alpha,lambda,epochs,false); 
    mlp.addExternalTestSet(validDS);
    mlp.training();
    cout << "Accuracy is " << mlp.getAccuracy() << endl;
    cout << "Error is " << mlp.getMEEValue() << endl;
    mlp.savePlotName(monkName);
    sleep(1);

}


