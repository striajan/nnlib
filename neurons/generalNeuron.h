#ifndef _GENERAL_NEURON_H_
#define _GENERAL_NEURON_H_

#include "neuronBase.h"
#include "../activationFunctions/activationFunc.h"
#include "../combinators/combinator.h"

namespace NNLib
{

	/**
	General neuron that can obtain any activation function and any combinator.
	*/
	template <typename T>
	class GeneralNeuron :
		public NeuronBase<T, ActivationFunc, Combinator>
	{
	public:
		GeneralNeuron(size_t inputsCount,
			const ActivationFunc<T> *activationFunc,
			const Combinator<T> *combinator) :
		NeuronBase(inputsCount, activationFunc, combinator)
		{ }
	};

}

#endif