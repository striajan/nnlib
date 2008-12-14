#include <iostream>
#include "common/range.h"
#include "common/random.h"
#include "activationFunctions/sigmoidFunc.h"
#include "activationFunctions/heavisideStepFunc.h"
#include "activationFunctions/symmetricSigmoidFunc.h"
#include "activationFunctions/tabbedSigmoidFunc.h"
#include "combinators/dotProduct.h"
#include "combinators/dotProductSSE.h"
#include "neurons/neuronBase.h"
#include "neurons/perceptron.h"
#include "feedForward/feedForwardLayer.h"
#include "feedForward/feedForwardNetwork.h"
#include "initializers/randomInitializer.h"
#include "data/inOutData.h"

using namespace NNLib;
using std::cout;
using std::endl;
using std::exception;

int main(int, char *[])
{
	// common
	const size_t INPUTS_COUNT = 4;
	const size_t H1_COUNT = 2;
	const size_t OUTPUTS_COUNT = 1;
	const Range<float> RANGE(-1, 1);
	const RandomUniform<float> RAND_GEN(RANGE);
	const RandomInitializer<float> RAND_INIT(RAND_GEN);
	const float INPUTS[INPUTS_COUNT] = {1,1,1,1};

	// activation functions
	SigmoidFunc<float> sigm;
	cout << sigm(0) << endl;
	HeavisideStepFunc<float> heav;
	cout << heav(0) << endl;
	SymmetricSigmoidFunc<float> sym;
	cout << sym(0) << endl;
	
	// tabbed sigmoid
	TabbedSigmoidFunc<float>::init(1.0f, 0.01f);
	const float SIGM_TEST[7] = {-3.004f, -2.0009f, -1.09f, 0, 1, 2, 3};
	TabbedSigmoidFunc<float> tabSigm;
	for (size_t i = 0; i < 7; ++i)
		cout << "(" << sigm(SIGM_TEST[i]) << "," << tabSigm(SIGM_TEST[i]) << ") ";
	cout << endl;

	// neuron 1 - base with symmetric sigmoid and dot product
	NeuronBase<float, SymmetricSigmoidFunc, DotProduct> n1(INPUTS_COUNT);
	try { n1.getWeight(INPUTS_COUNT); }
	catch (exception& ex) { cout << ex.what() << endl; }

	// neuron 2 - classic with sigmoid and dot product
	NeuronBase<float, SigmoidFunc, DotProduct> n2(INPUTS_COUNT);

	// random
	Random<float>::reset();
	RandomUniform<float> random(RANGE);
	cout << random.next() << endl;

	// perceptron - heaviside step function and dot product
	Perceptron<float> perc(INPUTS_COUNT);
	perc.initWeightsUniform(RANGE);
	float res = perc.eval(INPUTS);
	cout << res << endl;

	// feed-forward network typedefs
	typedef NeuronBase<float, TabbedSigmoidFunc, DotProductSSE> Neuron;
	typedef FeedForwardLayer<Neuron> Layer;
	typedef FeedForwardNetwork<Layer> Network;

	// network creation and initialization
	Network::LayersSizes sizes(2);
	sizes[0] = H1_COUNT;
	sizes[1] = OUTPUTS_COUNT;
	Network net(INPUTS_COUNT, sizes);
	net.initWeights(RAND_INIT);

	cout << net.getInputsCount() << " " << net.getOutputsCount() << " " <<
		net.getLayersCount() << endl;
	
	const Network::OutputType *out = net.eval(INPUTS);
	for (size_t i = 0; i < net.getOutputsCount(); ++i)
		cout << out[i] << " ";
	cout << endl;

	out = net.getOutputCache();
	for (size_t i = 0; i < net.getOutputsCount(); ++i)
		cout << out[i] << " ";
	cout << endl;
	
	TabbedSigmoidFunc<float>::finish();

	return 0;
}