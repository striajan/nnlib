#ifndef _DERIVABLE_ACTIVATION_FUNC_H_
#define _DERIVABLE_ACTIVATION_FUNC_H_

#include "activationFunc.h"

namespace NNLib
{

	/**
	Interface for an activation function that can be derived.
	*/
	template <typename T, typename R = T>
	class DerivableActivationFunc :
		public ActivationFunc<T, R>
	{
	public:
		/** Evaluate derivation of the function for the given 'x'. */
		virtual R derivation(T x) const = 0;
	};

}

#endif