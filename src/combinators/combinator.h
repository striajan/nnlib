#ifndef _COMBINATOR_H_
#define _COMBINATOR_H_

#include "combinators/combinatorBase.h"

namespace NNLib
{

	/**
	Common combinator - a function that takes two arrays of the same lengths
	and combines values stored in them.
	*/
	template <typename T>
	class Combinator :
		public CombinatorBase<T>
	{
	public:
		/** Combine values from two arrays of the given length. */
		virtual OutputType combine(const InputType[], const InputType[], size_t) const = 0;

		inline OutputType operator()(const InputType x[], const InputType y[], size_t len)
		{
			return combine(x, y, len);
		}

		virtual ~Combinator() = 0;
	};
	
	template <typename T>
	Combinator<T>::~Combinator()
	{ }

}

#endif