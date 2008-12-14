#ifndef _SYMMETRIC_SIGMOID_FUNC_H_
#define _SYMMETRIC_SIGMOID_FUNC_H_

#include <cmath>
#include "activationFunctions/activationFuncBase.h"
#include "activationFunctions/lambdaParamFunc.h"

namespace NNLib
{

	/**
	Symmetric sigmoid activation function.
	                 2
	f(x) = ---------------------- - 1
	        1 + exp(-lambda * x)
	*/
	template <typename T>
	class SymmetricSigmoidFunc :
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
		
		SymmetricSigmoidFunc(ParamType lambda = _LambdaParamFunc::DEF_LAMBDA_VAL) :
		_LambdaParamFunc(lambda)
		{}

		// interface ActivationFunc:

		ResultType function(ValueType x) const
		{
			return static_cast<ResultType>( 2 / (1 + ::exp(-this->m_lambda * x)) - 1 );
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
			return static_cast<ResultType>( 0.5 * this->m_lambda * (1 - y * y) );
		}
	};

}

#endif