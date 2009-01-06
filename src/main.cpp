#include <iostream>
#include <iomanip>
#include <fstream>
#include "common/range.h"
#include "common/random.h"
#include "activationFunctions/sigmoidFunc.h"
#include "combinators/dotProduct.h"
#include "neurons/neuronBase.h"
#include "feedForward/feedForwardLayer.h"
#include "feedForward/feedForwardNetwork.h"
#include "initializers/randomInitializer.h"
#include "data/inOutData.h"
#include "data/sequentialAccessor.h"
#include "data/iterCycleAccessor.h"
#include "data/dataOnNetworkStats.h"
#include "backPropagation/backPropBase.h"
#include "backPropagation/gradientEvaluator.h"
#include "backPropagation/weightsUpdater.h"
#include "backPropagation/continuator.h"
#include "backPropagation/distance.h"
#include "backPropagation/monitor.h"
#include "backPropagation/accumulator.h"

using namespace NNLib;

void identity()
{	
	// params
	const size_t INPUTS_COUNT  = 3;
	const size_t OUTPUTS_COUNT = 3;
	const size_t ITERS_COUNT  = 1;
	const size_t CYCLES_COUNT = 100000;
	const float LEARNING_RATE = 0.4f;
	const float MAX_ERROR = 0.005f;
	const size_t MONITOR_FREQUENCY = 99999;
	const char INPUT_FILE[] = "data/identity.in";

	// feed-forward network typedefs
	typedef NeuronBase<float, SigmoidFunc, DotProduct> Neuron;
	typedef FeedForwardLayer<Neuron> Layer;
	typedef FeedForwardNetwork<Layer> Network;

	// create layered network
	Network::LayersSizes sizes(3);
	sizes[0] = 2;
	sizes[1] = 5;
	sizes[2] = OUTPUTS_COUNT;
	Network net(INPUTS_COUNT, sizes);

	// init weights randomly
	net.initWeightsUniform( Range<float>(-1, 1) );

	// data initialization
	typedef InOutData< InOutPair<float> > TrainData;
	TrainData data(INPUTS_COUNT, OUTPUTS_COUNT);
	std::ifstream inputFile(INPUT_FILE);
	if ( inputFile.is_open() )
		data.load(inputFile);
	
	// data accessor creation
	typedef IterCycleAccessor<TrainData> ItAccessor;
	ItAccessor accessor(data, ITERS_COUNT, CYCLES_COUNT);

	// continuator
	typedef ErrorContinuator< Network, ItAccessor, MaxDistance,
		MaxAccumulator<float,8> > Continuator;
	Continuator continuator(net, accessor, MAX_ERROR);

	// monitors
	CombinedMonitor monitor;
	ParamMonitor<ItAccessor> m1(std::cout, accessor, MONITOR_FREQUENCY);
	monitor.add(m1);
	ParamMonitor<Continuator> m2(std::cout, continuator, MONITOR_FREQUENCY);
	monitor.add(m2);

	// back-propagation
	typedef BackPropBase<Network, DeltaGradientEvaluator,
		StandardUpdater> DeltaBarDelta;
	DeltaBarDelta back(net);
	back.setLearningRate(LEARNING_RATE);
	back.run(accessor, continuator, monitor);

	// test
	typedef SequentialAccessor<TrainData> SeqAccessor;
	SeqAccessor seq(data);
	typedef DataOnNetworkStats<Network, SeqAccessor, MaxDistance> Stats;
	Stats stats(net, seq);
	std::cout << std::endl << setiosflags(std::ios_base::fixed) <<
		std::setprecision(3);
	std::cout << "STATISTICS:\n" << stats << std::endl;
	//std::cout << "WEIGHTS:\n" << net << std::endl;
}

int main(int, char *[])
{
	identity();
	return 0;
}