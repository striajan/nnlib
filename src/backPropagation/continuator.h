#ifndef _CONTINUATOR_H_
#define _CONTINUATOR_H_

#include "initializers/constantInitializer.h"
#include <cstdlib>

namespace NNLib
{

	/**
	Base class for every class representing a continuator - a functor that returns
	whether the algorithm should continue (whatever it means).
	*/
	class ContinuatorBase
	{
	};


	/**
	This continuator always returns true so the algorithm should always continue.
	*/
	class AlwaysContinue :
		public ContinuatorBase
	{
	public:
		inline bool operator()() const { return true; }
	private:
		operator bool();
	};


	/**
	This continuator computes the average error on the network output over the given
	count of iterations. If this error is smaller than the given error it tells that
	no continuation is needed.
	*/
	template <typename NetworkT,
		typename DataAccessT,
		template <typename> class DistanceT>
	class AverageErrorContinuator :
		public ContinuatorBase,
		protected DistanceT<typename NetworkT::OutputType>
	{
	public:
		typedef NetworkT NetworkType;
		typedef typename NetworkType::OutputType OutputType;
		typedef typename NetworkType::OutputType ErrorType;
		typedef DistanceT<ErrorType> _DistanceBase;
		typedef DataAccessT DataAccessType;

		/** Create continuator for the given network and given data accessor. */
		AverageErrorContinuator(const NetworkType& network, const DataAccessType& accessor,
			ErrorType maxError, size_t memoryCapacity) :
		m_network(network), m_accessor(accessor), m_maxError(maxError), m_capacity(memoryCapacity)
		{
			m_memory = new ErrorType[m_capacity];
			resetMemory();
		}

		~AverageErrorContinuator()
		{
			delete [] m_memory;
		}

		bool operator()()
		{
			// move to the next position in history and increase size if needed
			m_pos = (m_pos + 1) % m_capacity;
			if (m_size < m_capacity)
				++m_size;

			// subtract the oldest error
			m_errorSum -= m_memory[m_pos];

			// compute the current error
			m_memory[m_pos] = distance( m_network.getOutputCache(), m_accessor.current().getOutput(),
				m_network.getOutputsCount() );
			
			// compute the average error
			m_errorSum += m_memory[m_pos];
			const ErrorType avgError = m_errorSum / static_cast<ErrorType>(m_size);

			return ( avgError > m_maxError );
		}

	protected:
		const NetworkType& m_network;
		const DataAccessType& m_accessor;

		/** If average error is smaller than this value the continuation stops. */
		const ErrorType m_maxError;

		/** Capacity of the memory from which the average error is being computed. */
		const size_t m_capacity;

		ErrorType *m_memory;
		ErrorType m_errorSum;
		size_t m_size;
		size_t m_pos;

		inline void resetMemory()
		{
			ConstantInitializer<ErrorType>(0)(m_memory, m_capacity);
			m_errorSum = 0;
			m_size = 0;
			m_pos = m_capacity - 1;
		}

	private:
		AverageErrorContinuator& operator=(const AverageErrorContinuator&);
	};

}

#endif