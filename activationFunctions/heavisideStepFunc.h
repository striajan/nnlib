#ifndef _HEAVISIDE_STEP_FUNC_H_
#define _HEAVISIDE_STEP_FUNC_H_

#include "activationFunc.h"

namespace NNLib
{

	/**
	Heaviside step activation function.
	f(x) = 0 ... for x < 0
	       1 ... for x >= 0
	*/
	template <typename T, typename R = T>
	class HeavisideStepFunc :
		public ActivationFunc<T, R>
	{
	public:
		// interface ActivationFunc:
		R function(T x) const
		{
			return static_cast<R>( (x < 0) ? 0 : 1 );
		}
	};

}

#endif