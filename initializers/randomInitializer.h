#ifndef _RANDOM_INITIALIZER_H_
#define _RANDOM_INITIALIZER_H_

#include "initializer.h"
#include "../common/random.h"

namespace NNLib
{

	/**
	Abstract initializer of an array of objects of the given type.
	*/
	template <typename T>
	class RandomInitializer :
		public Initializer<T>
	{
	public:
		typedef Random<T> RandomType;

		RandomInitializer(const RandomType& random = RandomUniform<T>) :
		m_random(random)
		{ }

		// interface Initializer:
		void operator()(ValType initWhat[], size_t len) const
		{
			for (size_t i = 0; i < len; ++i)
				initWhat[i] = m_random.next();
		}

	protected:
		/** Random numbers generator. */
		const RandomType& m_random;

		RandomInitializer& operator=(const RandomInitializer&);
	};

}

#endif