#ifndef _VAL_DERIVABLE_ACTIVATION_FUNC_H_
#define _VAL_DERIVABLE_ACTIVATION_FUNC_H_

#include "derivableActivationFunc.h"

namespace NNLib
{

	/**
	Interface for an activation function that can be derived according to the functional
	value. That means that for the given y = f(x) corresponding y' = f'(x) can be found.
	*/
	template <typename T, typename R = T>
	class ValDerivableActivationFunc :
		public DerivableActivationFunc<T, R>
	{
	public:
		/** Evaluate derivation of the function for the given functional value 'y'. */
		virtual R valDerivation(R y) const = 0;
	};

}

#endif