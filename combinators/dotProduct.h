#ifndef _DOT_PRODUCT_H_
#define _DOT_PRODUCT_H_

#include "combinator.h"

namespace NNLib
{

	/**
	Functor that computes dot product of two arrays.
	*/
	template <typename T>
	class DotProduct :
		public Combinator<T>
	{
	public:
		// interface Combinator:
		OutputType combine(const InputType x[], const InputType y[], size_t len) const
		{
			OutputType sum = 0;
			for (size_t i = 0; i < len; ++i)
				sum += x[i] * y[i];
			return sum;
		}
	};

}

#endif