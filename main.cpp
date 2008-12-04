#include <iostream>
#include "activationFunctions/sigmoidFunc.h"
#include "activationFunctions/heavisideStepFunc.h"
#include "activationFunctions/symmetricSigmoidFunc.h"
#include "combinators/dotProduct.h"
#include "neurons/neuronBase.h"
#include "neurons/generalNeuron.h"
#include "neurons/perceptron.h"
#include "data/inOutData.h"

using namespace NNLib;
using std::cout;
using std::endl;
using std::exception;

int main(int, char *[])
{
	// common
	const size_t INPUTS_COUNT = 4;
	const size_t OUTPUTS_COUNT = 1;
	const Range<float> RANGE(-1, 1);

	// activation functions
	SigmoidFunc<float> sigm;
	cout << sigm(0) << endl;
	HeavisideStepFunc<float> heav;
	cout << heav(0) << endl;
	SymmetricSigmoidFunc<float> sym;
	cout << sym(0) << endl;

	// combinators
	DotProduct<float> dot;

	// neuron 1 - base with symmetric sigmoid and dot product
	NeuronBase<float, SymmetricSigmoidFunc, DotProduct>
		n1(INPUTS_COUNT, &sym, &dot);
	//try { n1.getWeight(INPUTS_COUNT); }
	//catch (exception& ex) { cout << ex.what() << endl; }

	// neuron 2 - classic with sigmoid and dot product
	GeneralNeuron<float> n2(INPUTS_COUNT, &sigm, &dot);

	// random
	Random<float>::reset();
	RandomUniform<float> random(RANGE);
	cout << random.next() << endl;

	// perceptron - heaviside step function and dot product
	Perceptron<float> perc(INPUTS_COUNT, RANGE);
	float inputs[INPUTS_COUNT] = {1};
	float res = perc.eval(inputs);
	cout << res << endl;

	// patterns and expected outputs
	float in[INPUTS_COUNT], out[OUTPUTS_COUNT];
	InOutData<float> data(INPUTS_COUNT, OUTPUTS_COUNT);
	data.addCopy(in, out);

	return 0;
}