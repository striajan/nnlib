#ifndef _ERROR_EVALUATOR_H_
#define	_ERROR_EVALUATOR_H_

#include "backPropagation/accumulator.h"

namespace NNLib
{	
	
	template <typename NetworkT,
		typename DataAccessT,
		template <typename> class DistanceT>
	class DataOnNetworkStats :
		protected DistanceT<typename NetworkT::OutputType>
	{
	public:
		typedef NetworkT NetworkType;
		typedef typename NetworkType::OutputType ErrorType;
		typedef DataAccessT DataAccessType;

		DataOnNetworkStats(NetworkType& network, DataAccessType& accessor) :
		m_network(network), m_accessor(accessor)
		{ }

		void print(std::ostream& os)
		{
			static const std::string DELIM = "  ";
			
			const size_t inputsCount = m_network.getInputsCount();
			const size_t outputsCount = m_network.getOutputsCount();
			
			MeanAccumulator<ErrorType> mean;
			MaxAccumulator<ErrorType> maximum;
			SumAccumulator<ErrorType> sum;
			
			for ( m_accessor.begin(); !m_accessor.isEnd(); m_accessor.next() )
			{
				const typename DataAccessT::DataType& pattern = m_accessor.current();
				
				// eval output for the current input
				m_network.eval( pattern.getInput() );
				
				// compute current error
				ErrorType err = this->distance(m_network.getOutputCache(),
					pattern.getOutput(), outputsCount);
				
				print(os, pattern.getInput(), inputsCount);
				os << DELIM;
				print(os, pattern.getOutput(), outputsCount);
				os << DELIM;
				print(os, m_network.getOutputCache(), outputsCount);
				os << DELIM << err << std::endl;
				
				// accumulate errors
				maximum.accum(err);
				sum.accum(err);
				mean.accum(err);
			}
			
			os << "max=" << maximum.getAccumVal() << " sum=" << sum.getAccumVal() <<
				" mean=" << mean.getAccumVal() << std::endl;
		}

	protected:
		NetworkType& m_network;
		DataAccessType& m_accessor;
		
		template <typename T>
		inline void print(std::ostream& os, const T arr[], size_t len) const
		{
			for (size_t i = 0; i < len; ++i)
				os << arr[i] << " ";
		}

	private:
		DataOnNetworkStats& operator=(DataOnNetworkStats&);
	};
	
	
	template <typename N, typename A, template <typename> class D>
	inline std::ostream& operator<<(std::ostream& os, DataOnNetworkStats<N,A,D>& stat)
	{
		stat.print(os);
		return os;
	};
	
}


#endif