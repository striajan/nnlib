#ifndef _BACK_PROP_BASE_H_
#define	_BACK_PROP_BASE_H_

#include "feedForward/networkBufferAllocator.h"
#include "backPropagation/continuator.h"
#include "backPropagation/monitor.h"

namespace NNLib
{

	/**
	Back-propagation algorithm for a feed-forward layered network.
	*/
	template <typename NetworkT,
		template <typename> class WeightsStepsEvalT,
		template <typename> class WeightsUpdaterT>
	class BackPropBase :
		public WeightsStepsEvalT<NetworkT>,
		public WeightsUpdaterT<NetworkT>
	{
	public:
		typedef NetworkT NetworkType;
		typedef typename NetworkType::WeightType WeightType;
		typedef typename NetworkType::OutputType OutputType;
		typedef OutputType ErrorType;
		typedef WeightsStepsEvalT<NetworkType> WeightsStepsEvalType;
		typedef WeightsUpdaterT<NetworkType> WeightsUpdaterType;

		/** Init algorithm for the given network. */
		BackPropBase(NetworkType& network) :
		WeightsStepsEvalType(network),
		WeightsUpdaterType(network),
		m_network(network)
		{
			m_gradient = createWeightsBuffer<WeightType>(m_network);
		}

		~BackPropBase()
		{
			deleteWeightsBuffer(m_gradient);
		}

		template <typename DataAccessT>
		inline void run(DataAccessT& accessor)
		{
			AlwaysContinue continuator;
			EmptyMonitor monitor;
			return run(accessor, continuator, monitor);
		}

		template <typename DataAccessT, typename ContinuatorT>
		void run(DataAccessT& accessor, ContinuatorT& continuator, Monitor& monitor)
		{
			for ( accessor.begin(); !accessor.isEnd(); accessor.next() )
			{
				const typename DataAccessT::DataType& pattern = accessor.current();

				// get output for the given input
				m_network.eval( pattern.getInput() );

				// check the continuation condition
				if ( !continuator() )
					break;

				// run one step of the back-propagation algorithm
				evalGradient( pattern.getInput(), pattern.getOutput(), m_gradient );
				updateWeights( m_gradient );

				// monitor run of the back-propagation algorithm
				monitor();
			}
		}

		/** Run the back-propagation algorithm. */
		template <typename DataAccessT>
		inline void operator()(DataAccessT& accessor)
		{
			run(accessor);
		}

	protected:
		/** Network that should be trained. */
		NetworkType& m_network;

		/** Gradient of the error function (partial derivations of weights). */
		WeightType ***m_gradient;

	private:
		BackPropBase& operator=(const BackPropBase&);
	};

}

#endif