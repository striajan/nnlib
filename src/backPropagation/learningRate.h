#ifndef _LEARNING_RATE_H_
#define	_LEARNING_RATE_H_

#include "feedForward/networkBufferAllocator.h"

namespace NNLib
{
	
	/**
	Base class for for every learning rate class.
	*/
	template <typename T>
	class LearningRateBase
	{
	public:
		typedef T RateType;

	protected:
		static const RateType DEF_LEARNING_RATE;
	};

	template <typename T>
	const typename LearningRateBase<T>::RateType LearningRateBase<T>::DEF_LEARNING_RATE =
		static_cast<typename LearningRateBase<T>::RateType>( 0.2f );

	
	/**
	Global learning rate which is the same for the whole network.
	*/
	template <typename T>
	class GlobalLearningRate :
		public LearningRateBase<T>
	{
	private:
		typedef LearningRateBase<T> _LearningRateBase;
		
	public:
		typedef typename _LearningRateBase::RateType RateType;
		
		GlobalLearningRate(RateType rate = _LearningRateBase::DEF_LEARNING_RATE) :
		m_learningRate(rate)
		{ }
		
		inline RateType getLearningRate() const { return m_learningRate; }
		inline void setLearningRate(RateType rate) { m_learningRate = rate; }
		
	protected:		
		RateType m_learningRate;
	};
	
	
	/**
	Local learning rate which is unique for every single weight.
	*/
	template <typename T>
	class LocalLearningRate :
		public LearningRateBase<T>
	{
	private:
		typedef LearningRateBase<T> _LearningRateBase;
		
	public:
		typedef typename _LearningRateBase::RateType RateType;

		/** Create an array of local learning rates for the given network. */
		template <typename NetworkT>
		LocalLearningRate(const NetworkT& network)
		{
			m_learningRates = createWeightsBuffer<RateType>(network);
			for (size_t i = 0; i < network.getWeightsCount(); ++i)
				(**m_learningRates)[i] = _LearningRateBase::DEF_LEARNING_PARAM;
		}

		~LocalLearningRate()
		{
			deleteWeightsBuffer(m_learningRates);
		}
		
		inline RateType getLearningRate(size_t layer, size_t neuron, size_t input) const
		{
			return m_learningRates[layer][neuron][input];
		}
		
		inline void setLearningRate(size_t layer, size_t neuron, size_t input, RateType param)
		{
			m_learningRates[layer][neuron][input] = param;
		}
		
	protected:
		/** Local learning rates. */
		RateType ***m_learningRates;
	};
	
}

#endif