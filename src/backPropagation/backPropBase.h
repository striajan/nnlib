#ifndef _BACK_PROP_BASE_H_
#define	_BACK_PROP_BASE_H_

namespace NNLib
{

	/**
	Back-propagation algorithm for a feed-forward layered network.
	*/
	template <typename NetworkT>
	class BackPropBase
	{
	public:
		typedef NetworkT NetworkType;
		typedef typename NetworkType::OutputType OutputType;
		typedef OutputType ErrorType;
		
		BackPropBase(NetworkType& network) :
		m_network(network)
		{ }
		
		/*
		inline size_t getItersCount() const { return m_itersCount; }
		inline void setItersCount(size_t itersCount) const { m_itersCount = itersCount; }
		
		inline size_t getCyclesCount() const { return m_cyclesCount; }
		inline void setCyclesCount(size_t cyclesCount) const { m_cyclesCount = cyclesCount; }
		*/
		
	protected:
		/** Network that should be trained. */
		NetworkType& m_network;
		
		/** Number of iterations per one training pattern. */
		size_t m_itersCount;
		
		/** Number of cycles per all the training patterns. */
		size_t m_cyclesCount;
	};
	
}

#endif