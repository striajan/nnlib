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
	template <typename T>
	class SigmoidFunc :
		public ValDerivableActivationFunc<T>,
		public LambdaParamFunc<T, 2>
	{
	public:
		SigmoidFunc(ParamType lambda = DEF_LAMBDA_VAL) :
		LambdaParamFunc(lambda)
		{}

		// interface ActivationFunc:
		ResultType function(ValueType x) const
		{
			return static_cast<ResultType>( 1 / (1 + ::exp(-m_lambda * x)) );
		}

		// interface DerivableActivationFunc:
		ResultType derivation(ValueType x) const
		{
			return valDerivation( function(x) );
		}

		// interface ValDerivableActivationFunc:
		ResultType valDerivation(ResultType y) const
		{
			return static_cast<ResultType>( m_lambda * y * (1 - y) );
		}
	};

}

#endif