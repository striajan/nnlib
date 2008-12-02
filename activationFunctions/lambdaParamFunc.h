#ifndef _LAMBDA_PARAM_FUNC_
#define _LAMBDA_PARAM_FUNC_

namespace NNLib
{

	/**
	Function that uses some lambda as its parameter (whatever it means).
	*/
	template<typename P, int DEF_VAL = 0>
	class LambdaParamFunc
	{
	public:
		typedef P LambdaParamType;
		static const int DEF_LAMBDA_VAL = DEF_VAL;

		LambdaParamFunc(P lambda = DEF_LAMBDA_VAL) :
		m_lambda(lambda)
		{}

		inline P getLambda() const { return m_lambda; }
		inline void setLambda(P lambda) { m_lambda = lambda; }

	protected:
		/** Value of the lambda parameter. */
		P m_lambda;
	};

}

#endif