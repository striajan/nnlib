#ifndef _BACK_PROP_BASE_H_
#define	_BACK_PROP_BASE_H_

#include "feedForward/networkBufferAllocator.h"

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
			m_weightSteps = createWeightsBuffer<WeightType>(m_network);
		}

		~BackPropBase()
		{
			deleteWeightsBuffer(m_weightSteps);
		}

		/** Run the back-propagation algorithm. */
		template <typename DataAccessT>
		void run(DataAccessT& accessor)
		{
			typedef typename DataAccessT::DataType DataType;
			for ( accessor.begin(); !accessor.isEnd(); accessor.next() ) {
				const DataType& pattern = accessor.current();
				m_network.eval( pattern.getInput() );
				evalWeightsSteps( pattern.getInput(), pattern.getOutput(), m_weightSteps );
				updateWeights( m_weightSteps );
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

		/** Negative values of derivations for the weights. */
		WeightType ***m_weightSteps;

	private:
		BackPropBase& operator=(const BackPropBase&);
	};
	
}

#endif