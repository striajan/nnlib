#ifndef _LEARNING_RATE_H_
#define	_LEARNING_RATE_H_

#include "feedForward/networkBufferAllocator.h"
#include "initializers/constantInitializer.h"

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
		LocalLearningRate(const NetworkT& network, RateType rate = _LearningRateBase::DEF_LEARNING_RATE)
		{
			m_learningRates = createWeightsBuffer<RateType>(network);
			m_learningRatesLin = **m_learningRates;
			m_ratesCount = network.getWeightsCount();
			setLearningRate(rate);
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

		inline RateType getLearningRate(size_t index) const { return m_learningRatesLin[index]; }
		inline void setLearningRate(size_t index, RateType rate) { m_learningRatesLin[index] = rate; }

		inline void setLearningRate(RateType rate)
		{
			ConstantInitializer<RateType> init(rate);
			init(m_learningRatesLin, m_ratesCount);
		}
		
	protected:
		/** Local learning rates. */
		RateType ***m_learningRates;
		RateType *m_learningRatesLin;

		size_t m_ratesCount;
	};


	/**
	Adaptive learning rate.
	*/
	template <typename T>
	class AdaptiveRate
	{
	public:
		typedef T RateType;

		AdaptiveRate(RateType up, RateType down) :
		m_upRate(up), m_downRate(down)
		{ }

		inline RateType getUpRate() const { return m_upRate; }
		inline void setUpRate(RateType rate) { m_upRate = rate; }

		inline RateType getDownRate() const { return m_downRate; }
		inline void setDownRate(RateType rate) { m_downRate = rate; }

	protected:
		/** Up factor. */
		RateType m_upRate;

		/** Down factor. */
		RateType m_downRate;
	};


	/**
	Constraints for a learning rate.
	*/
	template <typename T>
	class MinMaxRate
	{
	public:
		typedef T RateType;

		MinMaxRate(RateType min, RateType max) :
		m_minRate(min), m_maxRate(max)
		{ }

		inline RateType getMinRate() const { return m_minRate; }
		inline void setMinRate(RateType rate) { m_minRate = rate; }

		inline RateType getMaxRate() const { return m_maxRate; }
		inline void setMaxRate(RateType rate) { m_maxRate = rate; }

	protected:
		/** Minimal learning rate. */
		RateType m_minRate;

		/** Maximal learning rate. */
		RateType m_maxRate;
	};
	
}

#endif