#ifndef _RANDOM_H_
#define _RANDOM_H_

#include <cstdlib>
#include <ctime>
#include "common/range.h"

namespace NNLib
{
	
	/**
	Base class for all random classes.
	*/
	template <typename T>
	class RandomBase
	{
	public:
		typedef T ResultType;
	};

	
	/**
	Ancestor for an every random numbers generator class.
	*/
	template <typename T>
	class Random :
		public RandomBase<T>
	{
	private:
		typedef RandomBase<T> _RandomBase;
			
	public:
		typedef typename _RandomBase::ResultType ResultType;

		virtual ~Random() = 0;

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
	
	template <typename T>
	Random<T>::~Random()
	{ }

	/**
	Choose random number from an uniform distribution.
	*/
	template <typename T>
	class RandomUniform :
		public Random<T>
	{
	private:
		typedef Random<T> _Random;
		
	public:
		typedef typename _Random::ResultType ResultType;
		typedef Range<T> RangeType;
		
		RandomUniform(const RangeType& range) :
		m_range(range)
		{ }
		
		~RandomUniform() { }

		// interface Random:
		ResultType next() const
		{
			return ( this->uniformRand() * m_range.getRange() + m_range.getMin() );
		}

	private:
		const RangeType m_range;

		RandomUniform& operator=(const RandomUniform&);
	};

}

#endif