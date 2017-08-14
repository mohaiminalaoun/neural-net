#include <iostream>
#include <vector>
#include "Net.h"

using namespace std;

int main() {



    //ctor

    // topology being 3, 2, 1 is 3 input neurons, 2 neurons in hidden layer and one output
    vector<unsigned> topology;



    //testing
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);

    Net myNet(topology);

    vector<double> inputVals;
    vector<double> targetVals;
    vector<double> resultVals;

    //training data
    myNet.feedForward(inputVals);
    //tell what outputs are supposed to be
    myNet.backProp(targetVals);
    //getting output
    myNet.getResults(resultVals);
}