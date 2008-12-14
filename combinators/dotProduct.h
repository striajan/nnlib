#ifndef _DOT_PRODUCT_H_
#define _DOT_PRODUCT_H_

#include "combinatorBase.h"

namespace NNLib
{

	/**
	Functor that computes dot product of two arrays.
	*/
	template <typename T>
	class DotProduct :
		public CombinatorBase<T>
	{
	private:
		typedef CombinatorBase<T> _CombinatorBase;
	
	public:
		typedef typename _CombinatorBase::InputType InputType;
		typedef typename _CombinatorBase::OutputType OutputType;
		
		// interface Combinator:

		OutputType combine(const InputType x[], const InputType y[], size_t len) const
		{
			OutputType sum = 0;
			for (size_t i = 0; i < len; ++i)
				sum += x[i] * y[i];
			return sum;
		}

		inline OutputType operator()(const InputType x[], const InputType y[], size_t len) const
		{
			return combine(x, y, len);
		}
	};

}

#endif