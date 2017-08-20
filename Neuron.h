//
// Created by Jesse on 8/18/2017.
//

#ifndef ML_NEURON_H
#define ML_NEURON_H


#include <cstdlib>
#include <vector>
#include "Net.h"

using namespace std;

class Neuron;

typedef vector<Neuron> Layer;

struct Connection {
    double weight;
    double deltaWeight;
};

class Neuron{
public:
    double mOutputVal;
    Neuron(unsigned numOutputs, unsigned mMyIndex);
    void feedForward(const Layer &previousLayer); // const as it doesn't need to modify the prev layer
    void setOutputVal(double val){ mOutputVal = val;} //define it in line since it's easy
    double getOutputVal() const{ return mOutputVal; }
    void calculateOutputGradients(double targetVal);
    void calculateHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &previousLayer);
private:
    //output val is the main private val

    vector<Connection> mOutputWeights; // weights to the right // has weight and delta weight from every neuron

    /*
     * Function to return a random weight
     */
    static double randomWeight(){
        return rand()/double(RAND_MAX);
    }

    unsigned mMyIndex;

    static double activationFunction(double x);
    static double activationFunctionDerivative(double x);

    double mGradient;

    double sumDOW(const Layer &nextLayer);

    static double eta;
    static double alpha;
};


#endif //ML_NEURON_H
