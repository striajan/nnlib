#ifndef _RANDOM_INITIALIZER_H_
#define _RANDOM_INITIALIZER_H_

#include "initializers/initializer.h"
#include "common/random.h"

namespace NNLib
{

	/**
	Random initializer of objects of the given type.
	*/
	template <typename T>
	class RandomInitializer :
		public Initializer<T>
	{
	private:
		typedef Initializer<T> _Initializer;
		
	public:
		typedef typename _Initializer::ValueType ValueType;
		typedef Random<T> RandomType;

		RandomInitializer(const RandomType& random) :
		m_random(random)
		{ }

		// interface Initializer:		
		void operator()(ValueType& initWhat) const
		{
			initWhat = m_random.next();
		}

	protected:
		/** Random numbers generator. */
		const RandomType& m_random;

		RandomInitializer& operator=(const RandomInitializer&);
	};

}

#endif