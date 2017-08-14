//
// Created by Jesse on 8/13/2017.
//

#include <iostream>
#include "Net.h"



// a Layer is a vector of neurons

Net::Net(const vector<unsigned> &topology){
    unsigned numLayers = (unsigned int) topology.size();
    for(unsigned layerNum = 0; layerNum<numLayers; ++layerNum){
        mLayers.push_back(Layer());
        // we do the loop +1 times because of the bias neuron
        for(unsigned neuronNum = 0; neuronNum<=topology[layerNum]; ++neuronNum){
            // .back is the last elem in the container
            mLayers.back().push_back(Neuron());
            cout<<"Created a Neuron"<<endl;
        }
    }
}

void Net::feedForward(const vector<double> &inputVals) {

}

void Net::backProp(const vector<double> targetVals) {

}

void Net::getResults(vector<double> &resultsVals) const {

}
