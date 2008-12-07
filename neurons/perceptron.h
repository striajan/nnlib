#ifndef _PERCEPTRON_H_
#define _PERCEPTRON_H_

#include "neuronBase.h"
#include "../common/range.h"
#include "../activationFunctions/heavisideStepFunc.h"
#include "../combinators/dotProduct.h"
#include "../initializers/randomInitializer.h"

namespace NNLib
{

	/**
	Perceptron - a neuron that uses heaviside step function and a dot product
	as a combinator.
	*/
	template <typename T>
	class Perceptron :
		public NeuronBase<T, HeavisideStepFunc, DotProduct>
	{
	public:
		Perceptron(size_t inputsCount) :
		NeuronBase(inputsCount)
		{}

		/** Set random weights from an uniform probability distribution. */
		void initWeightsUniform(const Range<T>& weightsRange)
		{
			RandomUniform<T> random(weightsRange);   // uniform random numbers generator
			RandomInitializer<T> init(random);       // weights initializer
			initWeights(init);
		}
	};

}

#endif
