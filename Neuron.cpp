

#include "Net.h"
#include "Neuron.h"

// give initial values to eta and alpha
double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

//creating ctr implementation here
Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
    // a neuron needs to know how many neurons in the next neuron it connects with
    for(unsigned c =0; c<numOutputs; c++){
        // every neuron has outward going arrows w/ weights
        mOutputWeights.push_back(Connection());
        //give random weight to each outward connection
        mOutputWeights.back().weight = randomWeight();
    }
    mMyIndex = myIndex;
}
// sum up the inputs, loop through all neurons in prev layer and accumulate the sum

void Neuron::feedForward(const Layer &previousLayer) {
    double sum = 0;
    // goes through all the neurons in the prev lauer
    for(unsigned n = 0; n<previousLayer.size(); ++n){
        sum += previousLayer[n].getOutputVal()*previousLayer[n].mOutputWeights[mMyIndex].weight;
    }

    mOutputVal = Neuron::activationFunction(sum);
    //
}

//Hyperbolic tangent functoin from -1 to 1
double Neuron::activationFunction(double x) {
    return tanh(x);
}

//approximation for the derivatce
double Neuron::activationFunctionDerivative(double x) {
    return 1-x*x;
}

void Neuron::calculateOutputGradients(double targetVal) {

    double delta = targetVal - mOutputVal;
    mGradient = delta * Neuron::activationFunction(mOutputVal);
}

void Neuron::calculateHiddenGradients(const Layer &nextLayer) {

    // same way as output gradients, only a little different

    double dow = sumDOW(nextLayer);
    mGradient = dow * Neuron::activationFunction(mOutputVal);
}

double Neuron::sumDOW(const Layer &nextLayer) {

    double sum = 0.0;

    // we need to sum contributions of the errors at the nodes we found

    for(unsigned n = 0; n<nextLayer.size()-1; n++){
        sum+=mOutputWeights[n].weight * nextLayer[n].mGradient;
    }

    return sum;



}

void Neuron::updateInputWeights(Layer &previousLayer) {

    // the weights that will be updated exist in the Connection container in the neurons in the previous layer

    for(unsigned n = 0; n < previousLayer.size(); n++){
        Neuron &neuron = previousLayer[n]; //get the neuron in the prev layer
        double oldDeltaWeight = neuron.mOutputWeights[mMyIndex].deltaWeight;
        // eta = learning rate
        double newDeltaWeight = eta * neuron.getOutputVal() * mGradient + alpha*oldDeltaWeight;

        neuron.mOutputWeights[mMyIndex].deltaWeight = newDeltaWeight;
        neuron.mOutputWeights[mMyIndex].weight += newDeltaWeight;
    }

}

