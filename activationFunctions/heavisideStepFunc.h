#ifndef _HEAVISIDE_STEP_FUNC_H_
#define _HEAVISIDE_STEP_FUNC_H_

#include "activationFuncBase.h"

namespace NNLib
{

	/**
	Heaviside step activation function.
	f(x) = 0 ... for x < 0
	       1 ... for x >= 0
	*/
	template <typename T>
	class HeavisideStepFunc :
		public ActivationFuncBase<T>
	{
	public:
		// interface ActivationFunc:

		ResultType function(ValueType x) const
		{
			return static_cast<ResultType>( (x < 0) ? 0 : 1 );
		}

		inline ResultType operator()(ValueType x)
		{
			return function(x);
		}
	};

}

#endif