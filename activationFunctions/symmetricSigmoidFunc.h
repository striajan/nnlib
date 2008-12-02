#ifndef _SYMMETRIC_SIGMOID_FUNC_H_
#define _SYMMETRIC_SIGMOID_FUNC_H_

#include <cmath>
#include "valDerivableActivationFunc.h"
#include "lambdaParamFunc.h"

namespace NNLib
{

	/**
	Symmetric sigmoid activation function.
	                 1
	f(x) = ---------------------- - 1
	        2 + exp(-lambda * x)
	*/
	template <typename T, typename R = T, typename P = T>
	class SymmetricSigmoidFunc :
		public ValDerivableActivationFunc<T, R>,
		public LambdaParamFunc<P, 1>
	{
	public:
		SymmetricSigmoidFunc(P lambda = LambdaParamFunc::DEF_LAMBDA_VAL) :
		LambdaParamFunc(lambda)
		{}

		// interface ActivationFunc:
		R function(T x) const
		{
			return static_cast<R>( 2 / (1 + ::exp(-m_lambda * x)) - 1 );
		}

		// interface DerivableActivationFunc:
		R derivation(T x) const
		{
			return valDerivation( function(x) );
		}

		// interface ValDerivableActivationFunc:
		R valDerivation(R y) const
		{
			return static_cast<R>( 0.5 * m_lambda * (1 - y * y) );
		}
	};

}

#endif