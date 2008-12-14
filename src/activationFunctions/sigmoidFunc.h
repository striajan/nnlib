#ifndef _SIGMOID_FUNC_H_
#define _SIGMOID_FUNC_H_

#include <cmath>
#include "activationFunctions/activationFuncBase.h"
#include "activationFunctions/lambdaParamFunc.h"

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
		public ActivationFuncBase<T>,
		public LambdaParamFunc<T, 1>
	{
	private:
		typedef ActivationFuncBase<T> _ActivationFuncBase;
		typedef LambdaParamFunc<T, 1> _LambdaParamFunc;
		
	public:
		typedef typename _ActivationFuncBase::ValueType ValueType;
		typedef typename _ActivationFuncBase::ResultType ResultType;
		typedef typename _LambdaParamFunc::ParamType ParamType;
		using _LambdaParamFunc::DEF_LAMBDA_VAL;
		
		SigmoidFunc(ParamType lambda = _LambdaParamFunc::DEF_LAMBDA_VAL) :
		_LambdaParamFunc(lambda)
		{}

		// interface ActivationFunc:

		ResultType function(ValueType x) const
		{
			return static_cast<ResultType>( 1 / (1 + ::exp(-this->m_lambda * x)) );
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
			return static_cast<ResultType>( this->m_lambda * y * (1 - y) );
		}
	};

}

#endif