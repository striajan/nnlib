#ifndef _CONTINUATOR_H_
#define _CONTINUATOR_H_

#include <cstdlib>
#include <ostream>

#ifdef UNIX
#	include <signal.h>
#endif

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
	This continuator computes the average error on the network's output. If this
	error is smaller than the given error it tells that no continuation is needed.
	*/
	template <typename NetworkT,
		typename DataAccessT,
		template <typename> class DistanceT,
		typename ErrorAccumT>
	class ErrorContinuator :
		public ContinuatorBase,
		protected DistanceT<typename NetworkT::OutputType>,
		protected ErrorAccumT
	{
	public:
		typedef NetworkT NetworkType;
		typedef typename NetworkType::OutputType ErrorType;
		typedef DataAccessT DataAccessType;

		/** Create continuator for the given network and given data accessor. */
		ErrorContinuator(const NetworkType& network, const DataAccessType& accessor, ErrorType maxError) :
		m_network(network), m_accessor(accessor), m_maxError(maxError)
		{ }

		bool operator()()
		{
			m_lastError = distance( m_network.getOutputCache(),
				m_accessor.current().getOutput(), m_network.getOutputsCount() );
			this->accum(m_lastError);
			return ( this->getAccumVal() > m_maxError );
		}

		inline ErrorType getMaxError() const { return m_maxError; }
		inline ErrorType getLastError() const { return m_lastError; }
		inline ErrorType getError() const { return this->getAccumVal(); }

	protected:
		const NetworkType& m_network;
		const DataAccessType& m_accessor;

		/** If an error is smaller than this value the continuation stops. */
		const ErrorType m_maxError;

		ErrorType m_lastError;

	private:
		ErrorContinuator& operator=(const ErrorContinuator&);
	};
	
	
	/** Print the error and the maximal error. */
	template <typename A, typename B, template <typename> class C, typename D>
	std::ostream& operator<<(std::ostream& os, const ErrorContinuator<A,B,C,D>& error)
	{
		os << "last=" << error.getLastError() << " overall=" << error.getError();
		return os;
	}
	
	
	/**
	Console input continuator which fails if exit sequence Ctrl+C was entered.
	*/
	class ConsoleInterruptionContinuator :
		public ContinuatorBase
	{
	public:
		ConsoleInterruptionContinuator()
		{
			#ifdef UNIX
				signal(SIGINT, signalHandler);
			#endif
			s_continue = true;
		}
		
		inline bool operator()() const
		{
			return s_continue;
		}
		
		static void signalHandler(int)
		{
			s_continue = false;
		}
		
	protected:
		static bool s_continue;
	};

}

#endif