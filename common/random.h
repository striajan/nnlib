#ifndef _RANDOM_H_
#define _RANDOM_H_

#include <cstdlib>
#include <ctime>
#include "../common/range.h"

namespace NNLib
{

	/**
	Ancestor for an every random numbers generator class.
	*/
	template <typename T>
	class Random
	{
	public:
		typedef T ResultType;

		virtual ~Random() = 0 { }

		/** Generate next random number. */
		virtual ResultType next() const = 0;

		inline ResultType operator()() const
		{
			return next();
		}

		/** Sets a random starting point. */
		static void reset()
		{
			::srand( static_cast<unsigned int>(::time(NULL)) );
		}

	protected:
		/** Return random number uniformly distributed between 0 and 1. */
		ResultType uniformRand() const
		{
			double r = static_cast<double>(::rand()) / static_cast<double>(RAND_MAX);
			return static_cast<ResultType>(r);
		}
	};


	/**
	Choose random number from an uniform distribution.
	*/
	template <typename T>
	class RandomUniform :
		public Random<T>
	{
	public:
		RandomUniform(const Range<T>& range) :
		m_range(range)
		{ }

		// interface Random:
		ResultType next() const
		{
			return uniformRand() * m_range.getRange() + m_range.getMin();
		}

	private:
		const Range<T> m_range;

		RandomUniform& operator=(const RandomUniform&);
	};

}

#endif