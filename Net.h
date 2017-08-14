//
// Created by Jesse on 8/13/2017.
//

#ifndef ML_NET_H
#define ML_NET_H


#include <vector>

using namespace std;

class Neuron{

};
typedef vector<Neuron> Layer;

class Net {

private:
    vector<Layer> mLayers;


public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> targetVals);
    void getResults(vector<double> &resultsVals) const;


};


#endif //ML_NET_H
