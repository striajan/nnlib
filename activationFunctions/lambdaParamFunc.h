#ifndef _LAMBDA_PARAM_FUNC_
#define _LAMBDA_PARAM_FUNC_

namespace NNLib
{

	/**
	Function that uses some lambda as its parameter (whatever it means).
	*/
	template<typename T, int DEF_VAL = 0>
	class LambdaParamFunc
	{
	public:
		typedef T ParamType;
		static const int DEF_LAMBDA_VAL = DEF_VAL;

		LambdaParamFunc(ParamType lambda = DEF_LAMBDA_VAL) :
		m_lambda(lambda)
		{}

		inline ParamType getLambda() const { return m_lambda; }
		inline void setLambda(ParamType lambda) { m_lambda = lambda; }

	protected:
		/** Value of the lambda parameter. */
		ParamType m_lambda;
	};

}

#endif