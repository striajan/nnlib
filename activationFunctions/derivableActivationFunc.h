#ifndef _DERIVABLE_ACTIVATION_FUNC_H_
#define _DERIVABLE_ACTIVATION_FUNC_H_

#include "activationFunc.h"

namespace NNLib
{

	/**
	Interface for an activation function that can be derived.
	*/
	template <typename T>
	class DerivableActivationFunc :
		public ActivationFunc<T>
	{
	public:
		/** Evaluate derivation of the function for the given 'x'. */
		virtual ResultType derivation(ValueType x) const = 0;
	};

}

#endif