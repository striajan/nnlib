#ifndef _SIGMOID_FUNC_H_
#define _SIGMOID_FUNC_H_

#include <cmath>
#include "valDerivableActivationFunc.h"
#include "lambdaParamFunc.h"

namespace NNLib
{

	/**
	Sigmoid activation function.
	                1
	f(x) = ----------------------
	        1 + exp(-lambda * x)
	*/
	template <typename T, typename R = T, typename P = T>
	class SigmoidFunc :
		public ValDerivableActivationFunc<T, R>,
		public LambdaParamFunc<P, 2>
	{
	public:
		SigmoidFunc(P lambda = LambdaParamFunc::DEF_LAMBDA_VAL) :
		LambdaParamFunc(lambda)
		{}

		// interface ActivationFunc:
		R function(T x) const
		{
			return static_cast<R>( 1 / (1 + ::exp(-m_lambda * x)) );
		}

		// interface DerivableActivationFunc:
		R derivation(T x) const
		{
			return valDerivation( function(x) );
		}

		// interface ValDerivableActivationFunc:
		R valDerivation(R y) const
		{
			return static_cast<R>( m_lambda * y * (1 - y) );
		}
	};

}

#endif