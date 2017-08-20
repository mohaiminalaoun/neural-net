//
// Created by Jesse on 8/13/2017.
//

#include <iostream>
#include <cassert>
#include "Net.h"
#include "Neuron.h"



// a Layer is a vector of neurons

Net::Net(const vector<unsigned> &topology){
    unsigned numLayers = (unsigned int) topology.size();
    for(unsigned layerNum = 0; layerNum<numLayers; ++layerNum){
        mLayers.push_back(Layer());

        unsigned numOutputs;
        // if last layer, output nums is 0
        if(layerNum==topology.size()-1){
            numOutputs = 0;
        }else{
            numOutputs = topology[layerNum+1]; //num of neurons in the next layer
        }
        // we do the loop +1 times because of the bias neuron
        for(unsigned neuronNum = 0; neuronNum<=topology[layerNum]; ++neuronNum){
            // .back is the last elem in the container
            // basically create # neurons per layer
            mLayers.back().push_back(Neuron(numOutputs,neuronNum));
            cout<<"Created a Neuron"<<endl;
        }
    }
}


//putting input to the neurons and pushing it to the next layer
void Net::feedForward(const vector<double> &inputVals) {

    //check
    assert(inputVals.size()==mLayers[0].size()-1);

    //assign input values to the input neurons
    for(unsigned i = 0; i < inputVals.size (); i++){
        mLayers[0][i].setOutputVal(inputVals[i]);
    }
    //now forward propagate
    // start from the next layer
    for(unsigned layerNum = 1; layerNum< mLayers.size(); ++layerNum){
        Layer &previousLayer = mLayers[layerNum-1]; // reference to previous layer //very fast operation, no copy
        //for each neuron in every layer
        for(unsigned n = 0; n< mLayers[layerNum].size()-1; ++n){
            mLayers[layerNum][n].feedForward(previousLayer); // method that does the math to update output value
            //neuron does the math
        }
    }
}

void Net::backProp(const vector<double> targetVals) {

    // Calculate overall net errors (RMS of output neuron erros)

    Layer &outputLayer = mLayers.back();

    mError = 0.0;

    for(unsigned i = 0; i < outputLayer.size() -1; i++){
        double delta = targetVals[i] - ((Neuron)outputLayer[i]).mOutputVal;
        mError += delta*delta;
    }
    mError = mError/(outputLayer.size()-1);
    mError = sqrt(mError);


    // calculate output layer gradients

    for(unsigned n = 0; n < outputLayer.size() -1; n++){
        outputLayer[n].calculateOutputGradients(targetVals[n]); //neuron calculates the gradient
    }

    for(unsigned layerNum = mLayers.size()-2; layerNum>0; --layerNum){
        Layer &hiddenLayer = mLayers[layerNum];
        Layer &nextLayer = mLayers[layerNum+1];

        for(unsigned n = 0; n <hiddenLayer.size(); n++){
            hiddenLayer[n].calculateHiddenGradients(nextLayer);
        }

    }


    // for all layers from output to first hidden layer, update connection weights

    for(unsigned layerNum = mLayers.size()-1; layerNum>0; --layerNum){
        Layer &layer = mLayers[layerNum];
        Layer &prevLayer = mLayers[layerNum-1];

        for(unsigned n = 0; n <layer.size(); n++){
            layer[n].updateInputWeights(prevLayer);

        }

    }





}

void Net::getResults(vector<double> &resultsVals) const {

    resultsVals.clear(); //clear container passed to it
    for(unsigned n = 0; n < mLayers.back().size(); n++){
        resultsVals.push_back(mLayers.back()[n].getOutputVal());
    }
}
