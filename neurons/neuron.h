#ifndef _NEURON_H_
#define _NEURON_H_

#include "neuronBase.h"
#include "../activationFunctions/activationFunc.h"
#include "../combinators/combinator.h"

namespace NNLib
{

	/**
	General that can obtain any activation function and any combinator.
	*/
	template <typename T>
	class Neuron :
		public NeuronBase<T, ActivationFunc, Combinator>
	{
	public:
		Neuron(size_t inputsCount,
			const ActivationFunc<T> *activationFunc,
			const Combinator<T> *combinator) :
		NeuronBase(inputsCount, activationFunc, combinator)
		{ }
	};

}

#endif