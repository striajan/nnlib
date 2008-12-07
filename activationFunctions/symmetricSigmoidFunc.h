#ifndef _SYMMETRIC_SIGMOID_FUNC_H_
#define _SYMMETRIC_SIGMOID_FUNC_H_

#include <cmath>
#include "activationFuncBase.h"
#include "lambdaParamFunc.h"

namespace NNLib
{

	/**
	Symmetric sigmoid activation function.
	                 1
	f(x) = ---------------------- - 1
	        2 + exp(-lambda * x)
	*/
	template <typename T>
	class SymmetricSigmoidFunc :
		public ActivationFuncBase<T>,
		public LambdaParamFunc<T, 1>
	{
	public:
		SymmetricSigmoidFunc(ParamType lambda = DEF_LAMBDA_VAL) :
		LambdaParamFunc(lambda)
		{}

		// interface ActivationFunc:

		ResultType function(ValueType x) const
		{
			return static_cast<ResultType>( 2 / (1 + ::exp(-m_lambda * x)) - 1 );
		}

		inline ResultType operator()(ValueType x)
		{
			return function(x);
		}

		// interface DerivableActivationFunc:

		ResultType derivation(ValueType x) const
		{
			return valDerivation( function(x) );
		}

		// interface ValDerivableActivationFunc:

		ResultType valDerivation(ResultType y) const
		{
			return static_cast<ResultType>( 0.5 * m_lambda * (1 - y * y) );
		}
	};

}

#endif