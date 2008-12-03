#include <iostream>

#include "activationFunctions/sigmoidFunc.h"
#include "activationFunctions/heavisideStepFunc.h"
#include "activationFunctions/symmetricSigmoidFunc.h"
#include "neurons/neuron.h"

using namespace NNLib;
using std::cout;
using std::endl;
using std::exception;

int main(int, char *[])
{
	// activation functions

	SigmoidFunc<float> sigm;
	cout << sigm(0) << endl;

	HeavisideStepFunc<float> heav;
	cout << heav(0) << endl;

	SymmetricSigmoidFunc<float> sym;
	cout << sym(0) << endl;

	// combinators

	DotProductCombinator<float> dot;

	// neurons

	const size_t INPUTS_COUNT = 8;

	NeuronBase<float> n1(INPUTS_COUNT, sigm, dot);

	try {
		n1.getWeight(INPUTS_COUNT);
	}
	catch (exception& ex) {
		cout << ex.what() << endl;
	}

	Neuron<float> n2(INPUTS_COUNT, heav, dot);

	return 0;
}