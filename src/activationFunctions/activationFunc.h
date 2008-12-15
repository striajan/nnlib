#ifndef _ACTIVATION_FUNC_H_
#define _ACTIVATION_FUNC_H_

#include "activationFunctions/activationFuncBase.h"

namespace NNLib
{

	/**
	Interface for a common activation function of a neuron.
	*/
	template <typename T>
	class ActivationFunc :
		public ActivationFuncBase<T>
	{
	public:
		/** Evaluate the function for the given 'x'. */
		virtual ResultType function(ValueType x) const = 0;

		/** Evaluate the function for the given 'x'. */
		inline ResultType operator()(ValueType x) const
		{
			return function(x);
		}

		virtual ~ActivationFunc() = 0;
	};
	
	template <typename T>
	ActivationFunc<T>::~ActivationFunc()
	{ }

}

#endif