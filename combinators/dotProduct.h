#ifndef _DOT_PRODUCT_COMBINATOR_
#define _DOT_PRODUCT_COMBINATOR_

#include "combinator.h"

namespace NNLib
{

	/**
	Functor that computes dot product of two arrays.
	*/
	template <typename T>
	class DotProductCombinator :
		public Combinator<T>
	{
	public:
		// interface Combinator:
		virtual OutputType operator()(const InputType x[], const InputType y[], size_t len) const
		{
			OutputType sum = 0;
			for (size_t i = 0; i < len; ++i)
				sum += x[i] + y[i];
			return sum;
		}
	};

}

#endif