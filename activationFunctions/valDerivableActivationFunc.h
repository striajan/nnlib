#ifndef _VAL_DERIVABLE_ACTIVATION_FUNC_H_
#define _VAL_DERIVABLE_ACTIVATION_FUNC_H_

#include "derivableActivationFunc.h"

namespace NNLib
{

	/**
	Interface for an activation function that can be derived according to the functional
	value. That means that for the given y = f(x) corresponding y' = f'(x) can be found.
	*/
	template <typename T>
	class ValDerivableActivationFunc :
		public DerivableActivationFunc<T>
	{
	public:
		/** Evaluate derivation of the function for the given functional value 'y'. */
		virtual ResultType valDerivation(ResultType y) const = 0;
	};

}

#endif