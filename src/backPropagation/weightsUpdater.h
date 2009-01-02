#ifndef _WEIGHTS_UPDATER_H_
#define _WEIGHTS_UPDATER_H_

#include "backPropagation/learningRate.h"
#include "backPropagation/learningMomentum.h"
#include "initializers/constantInitializer.h"

namespace NNLib
{

// BASE CLASS /////////////////////////////////////////////////////////////////

	/**
	Base class for every updater of a neural network's weights.
	*/
	template <typename NetworkT>
	class WeightsUpdaterBase
	{
	public:
		typedef NetworkT NetworkType;
		typedef typename NetworkType::WeightType WeightType;

		WeightsUpdaterBase(NetworkType& network) :
		m_network(network)
		{ }

	protected:
		/** Network which weights should be updated. */
		NetworkType& m_network;

	private:
		WeightsUpdaterBase& operator=(const WeightsUpdaterBase&);
	};


// STANDARD BACK-PROPAGATION ALGORITHM ////////////////////////////////////////

	/**
	Standard version of weights updater with a global learning rate and momentum.
	*/
	template <typename NetworkT>
	class StandardUpdater :
		public WeightsUpdaterBase<NetworkT>,
		public GlobalLearningRate<typename NetworkT::WeightType>,
		public GlobalLearningMomentum<typename NetworkT::WeightType>
	{
	private:
		typedef WeightsUpdaterBase<NetworkT> _UpdaterBase;
		typedef GlobalLearningRate<typename NetworkT::WeightType> _LearningRateBase;
		typedef GlobalLearningMomentum<typename NetworkT::WeightType> _LearningMomentumBase;

	public:
		typedef typename _UpdaterBase::NetworkType NetworkType;
		typedef typename _UpdaterBase::WeightType WeightType;
		typedef typename _LearningRateBase::RateType RateType;
		typedef typename _LearningMomentumBase::MomentumType MomentumType;
		
		StandardUpdater(NetworkType& network) :
		_UpdaterBase(network)
		{
			m_weightsCache = new WeightType[network.getWeightsCount()];
			reset();
		}

		~StandardUpdater()
		{
			delete [] m_weightsCache;
		}

		void updateWeights(WeightType ***gradient)
		{
			m_gradient = **gradient;
			m_index = 0;
			this->m_network.forEachWeightForward( *this );
		}
		
		/** Update one given weight. */
		void operator()(WeightType& weight)
		{
			// cache the old weight
			WeightType oldWeight = m_weightsCache[m_index];
			m_weightsCache[m_index] = weight;

			// update the weight
			weight += this->getLearningMomentum() * ( m_weightsCache[m_index] - oldWeight ) -
				this->getLearningRate() * m_gradient[m_index];

			// move to the next weight
			++m_index;
		}

		void reset()
		{
			ConstantInitializer<WeightType>(0)
				( m_weightsCache, this->m_network.getWeightsCount() );
		}

	protected:
		WeightType *m_weightsCache;
		WeightType *m_gradient;
		size_t m_index;
	};


// SILVA AND ALMEIDA'S ALGORITHM //////////////////////////////////////////////

	/**
	Updater used by Silva and Almeida's algorithm.
	*/
	template <typename NetworkT>
	class SilvaAlmeidaUpdater :
		public WeightsUpdaterBase<NetworkT>,
		public LocalLearningRate<typename NetworkT::WeightType>,
		public AdaptiveRate<typename NetworkT::WeightType>
	{
	private:
		typedef WeightsUpdaterBase<NetworkT> _UpdaterBase;
		typedef LocalLearningRate<typename NetworkT::WeightType> _LearningRateBase;
		typedef AdaptiveRate<typename NetworkT::WeightType> _AdaptiveRateBase;

	public:
		typedef typename _UpdaterBase::NetworkType NetworkType;
		typedef typename _UpdaterBase::WeightType WeightType;
		typedef typename _LearningRateBase::RateType RateType;

		SilvaAlmeidaUpdater(NetworkType& network, RateType up = DEF_UP_RATE, RateType down = DEF_DOWN_RATE) :
		_UpdaterBase(network), _LearningRateBase(network), _AdaptiveRateBase(up, down)
		{
			m_gradientCache = new WeightType[network.getWeightsCount()];
			reset();
		}

		~SilvaAlmeidaUpdater()
		{
			delete [] m_gradientCache;
		}

		void updateWeights(WeightType ***gradient)
		{
			m_gradient = **gradient;
			m_index = 0;
			this->m_network.forEachWeightForward( *this );
		}
		
		/** Update one given weight. */
		void operator()(WeightType& weight)
		{
			static const WeightType ZERO_WEIGHT = static_cast<WeightType>(0);

			// update the weight
			weight -= this->getLearningRate(m_index) * m_gradient[m_index];

			// update the learning rate
			WeightType signum = m_gradientCache[m_index] * m_gradient[m_index];
			if (signum > ZERO_WEIGHT)
				this->setLearningRate( m_index, this->getLearningRate(m_index) * this->getUpRate() );
			else if (signum < ZERO_WEIGHT)
				this->setLearningRate( m_index, this->getLearningRate(m_index) * this->getDownRate() );

			// cache the gradient
			m_gradientCache[m_index] = m_gradient[m_index];

			// move to the next weight
			++m_index;
		}

		void reset()
		{
			ConstantInitializer<WeightType>(0)
				( m_gradientCache, this->m_network.getWeightsCount() );
		}

	protected:
		WeightType *m_gradientCache;
		WeightType *m_gradient;
		size_t m_index;

		static const RateType DEF_UP_RATE;
		static const RateType DEF_DOWN_RATE;
	};

	template <typename NetworkT>
	const typename SilvaAlmeidaUpdater<NetworkT>::RateType SilvaAlmeidaUpdater<NetworkT>::DEF_UP_RATE
		= static_cast<typename SilvaAlmeidaUpdater<NetworkT>::RateType>( 1.2f );

	template <typename NetworkT>
	const typename SilvaAlmeidaUpdater<NetworkT>::RateType SilvaAlmeidaUpdater<NetworkT>::DEF_DOWN_RATE
		= static_cast<typename SilvaAlmeidaUpdater<NetworkT>::RateType>( 0.8f );


// DELTA-BAR-DELTA ALGORITHM //////////////////////////////////////////////////

	/**
	Updater used by Delta-bar-delta algorithm.
	*/
	template <typename NetworkT>
	class DeltaBarDeltaUpdater :
		public WeightsUpdaterBase<NetworkT>,
		public LocalLearningRate<typename NetworkT::WeightType>,
		public AdaptiveRate<typename NetworkT::WeightType>
	{
	private:
		typedef WeightsUpdaterBase<NetworkT> _UpdaterBase;
		typedef LocalLearningRate<typename NetworkT::WeightType> _LearningRateBase;
		typedef AdaptiveRate<typename NetworkT::WeightType> _AdaptiveRateBase;

	public:
		typedef typename _UpdaterBase::NetworkType NetworkType;
		typedef typename _UpdaterBase::WeightType WeightType;
		typedef typename _LearningRateBase::RateType RateType;

		DeltaBarDeltaUpdater(NetworkType& network, RateType up = DEF_UP_RATE, RateType down = DEF_DOWN_RATE,
		RateType inertia = DEF_INERTIA) :
		_UpdaterBase(network), _LearningRateBase(network), _AdaptiveRateBase(up, down)
		{
			m_deltasCache = new WeightType[network.getWeightsCount()];
			setInertia(inertia);
			reset();
		}

		~DeltaBarDeltaUpdater()
		{
			delete [] m_deltasCache;
		}

		void updateWeights(WeightType ***gradient)
		{
			m_gradient = **gradient;
			m_index = 0;
			this->m_network.forEachWeightForward( *this );
		}
		
		/** Update one given weight. */
		void operator()(WeightType& weight)
		{
			static const WeightType ZERO_WEIGHT = static_cast<WeightType>(0);

			// update the weight
			weight -= this->getLearningRate(m_index) * m_gradient[m_index];

			// update the learning rate
			WeightType signum = m_gradient[m_index] * m_deltasCache[m_index];
			if (signum > ZERO_WEIGHT)
				this->setLearningRate( m_index, this->getLearningRate(m_index) + this->getUpRate() );
			else if (signum < ZERO_WEIGHT)
				this->setLearningRate( m_index, this->getLearningRate(m_index) * this->getDownRate() );

			// update the delta param
			m_deltasCache[m_index] = getInertiaInv() * m_gradient[m_index] +
				getInertia() * m_deltasCache[m_index];

			// move to the next weight
			++m_index;
		}

		void reset()
		{
			ConstantInitializer<WeightType>(0)
				( m_deltasCache, this->m_network.getWeightsCount() );
		}

		inline RateType getInertia() const { return m_inertia; }
		inline RateType getInertiaInv() const { return m_inertiaInv; }
		
		inline void setInertia(RateType inertia)
		{
			m_inertia = inertia;
			m_inertiaInv = static_cast<RateType>(1) - inertia;
		}

	protected:
		WeightType *m_deltasCache;
		WeightType *m_gradient;
		size_t m_index;
		WeightType m_inertia, m_inertiaInv;

		static const RateType DEF_UP_RATE;
		static const RateType DEF_DOWN_RATE;
		static const WeightType DEF_INERTIA;
	};

	template <typename NetworkT>
	const typename DeltaBarDeltaUpdater<NetworkT>::RateType DeltaBarDeltaUpdater<NetworkT>::DEF_UP_RATE
		= static_cast<typename DeltaBarDeltaUpdater<NetworkT>::RateType>( 0.09f );

	template <typename NetworkT>
	const typename DeltaBarDeltaUpdater<NetworkT>::RateType DeltaBarDeltaUpdater<NetworkT>::DEF_DOWN_RATE
		= static_cast<typename DeltaBarDeltaUpdater<NetworkT>::RateType>( 0.8f );

	template <typename NetworkT>
	const typename DeltaBarDeltaUpdater<NetworkT>::WeightType DeltaBarDeltaUpdater<NetworkT>::DEF_INERTIA
		= static_cast<typename DeltaBarDeltaUpdater<NetworkT>::WeightType>( 0.5f );


// SUPER SAB ALGORITHM ////////////////////////////////////////////////////////

	/**
	Updater used by Super SAB (Self-Adapting Back-propagation) algorithm.
	*/
	template <typename NetworkT>
	class SuperSABUpdater :
		public WeightsUpdaterBase<NetworkT>,
		public LocalLearningRate<typename NetworkT::WeightType>,
		public AdaptiveRate<typename NetworkT::WeightType>,
		public GlobalLearningMomentum<typename NetworkT::WeightType>
	{
	private:
		typedef WeightsUpdaterBase<NetworkT> _UpdaterBase;
		typedef LocalLearningRate<typename NetworkT::WeightType> _LearningRateBase;
		typedef AdaptiveRate<typename NetworkT::WeightType> _AdaptiveRateBase;
		typedef GlobalLearningMomentum<typename NetworkT::WeightType> _LearningMomentumBase;

	public:
		typedef typename _UpdaterBase::NetworkType NetworkType;
		typedef typename _UpdaterBase::WeightType WeightType;
		typedef typename _LearningRateBase::RateType RateType;
		typedef typename _LearningMomentumBase::MomentumType MomentumType;

		SuperSABUpdater(NetworkType& network, RateType up = DEF_UP_RATE, RateType down = DEF_DOWN_RATE) :
		_UpdaterBase(network), _LearningRateBase(network), _AdaptiveRateBase(up, down)
		{
			size_t weightsCount = network.getWeightsCount();
			m_gradientCache = new WeightType[weightsCount];
			m_weightsCache  = new WeightType[weightsCount];
			m_stepsCache  = new WeightType[weightsCount];
			reset();
		}

		~SuperSABUpdater()
		{
			delete [] m_gradientCache;
			delete [] m_weightsCache;
			delete [] m_stepsCache;
		}

		void updateWeights(WeightType ***gradient)
		{
			m_gradient = **gradient;
			m_index = 0;
			this->m_network.forEachWeightForward( *this );
		}
		
		/** Update one given weight. */
		void operator()(WeightType& weight)
		{
			static const WeightType ZERO_WEIGHT = static_cast<WeightType>(0);

			// change of derivation
			WeightType signum = m_gradientCache[m_index] * m_gradient[m_index];
			
			if (signum >= ZERO_WEIGHT)
			{
				// compute current step and cache it
				m_stepsCache[m_index] = this->getLearningMomentum() * ( weight - m_weightsCache[m_index] ) -
					this->getLearningRate(m_index) * m_gradient[m_index];

				// cache the old weight and the old gradient
				m_weightsCache[m_index] = weight;
				m_gradientCache[m_index] = m_gradient[m_index];

				// update the weight
				weight += m_stepsCache[m_index];

				// update the learning rate if needed
				RateType learningRate = this->getLearningRate(m_index);
				if ( learningRate < getMaxLearningRate() )
					this->setLearningRate( m_index, learningRate * this->getUpRate() );
			}
			else
			{
				// reset the last weight update
				weight -= m_stepsCache[m_index];

				// annulate the step cache and the update cache
				m_stepsCache[m_index] = m_gradientCache[m_index] = ZERO_WEIGHT;

				// update the learning rate
				this->setLearningRate( m_index, this->getLearningRate(m_index) * this->getDownRate() );
			}

			// move to the next weight
			++m_index;
		}

		void reset()
		{
			ConstantInitializer<WeightType> init(0);
			size_t weightsCount = this->m_network.getWeightsCount();
			init(m_gradientCache, weightsCount);
			init(m_weightsCache, weightsCount);
			init(m_stepsCache, weightsCount);
		}

		inline WeightType getMaxLearningRate() const { return m_maxLearningRate; }
		inline void setMaxLearningRate(WeightType rate) { m_maxLearningRate = rate; }

	protected:
		WeightType *m_gradientCache;
		WeightType *m_weightsCache;
		WeightType *m_stepsCache;
		WeightType *m_gradient;
		WeightType m_maxLearningRate;
		size_t m_index;

		static const RateType DEF_UP_RATE;
		static const RateType DEF_DOWN_RATE;
		static const RateType DEF_MAX_RATE;
	};

	template <typename NetworkT>
	const typename SuperSABUpdater<NetworkT>::RateType SuperSABUpdater<NetworkT>::DEF_UP_RATE
		= static_cast<typename SuperSABUpdater<NetworkT>::RateType>( 1.05f );

	template <typename NetworkT>
	const typename SuperSABUpdater<NetworkT>::RateType SuperSABUpdater<NetworkT>::DEF_DOWN_RATE
		= static_cast<typename SuperSABUpdater<NetworkT>::RateType>( 0.5f );

	template <typename NetworkT>
	const typename SuperSABUpdater<NetworkT>::RateType SuperSABUpdater<NetworkT>::DEF_MAX_RATE
		= static_cast<typename SuperSABUpdater<NetworkT>::RateType>( 0.8f );


// QUICKPROP ALGORITHM ////////////////////////////////////////////////////////

	/**
	Updater used by the Quickprop (Quick-propagation) algorithm.
	*/
	template <typename NetworkT>
	class QuickpropUpdater :
		public WeightsUpdaterBase<NetworkT>
	{
	private:
		typedef WeightsUpdaterBase<NetworkT> _UpdaterBase;

	public:
		typedef typename _UpdaterBase::NetworkType NetworkType;
		typedef typename _UpdaterBase::WeightType WeightType;
		
		QuickpropUpdater(NetworkType& network) :
		_UpdaterBase(network)
		{
			size_t weightsCount = network.getWeightsCount();
			m_gradientCache = new WeightType[weightsCount];
			m_stepsCache = new WeightType[weightsCount];
			reset();
		}

		~QuickpropUpdater()
		{
			delete [] m_gradientCache;
			delete [] m_stepsCache;
		}

		void updateWeights(WeightType ***gradient)
		{
			m_gradient = **gradient;
			m_index = 0;
			this->m_network.forEachWeightForward( *this );
		}
		
		/** Update one given weight. */
		void operator()(WeightType& weight)
		{
			// difference between current and previous gradient
			const WeightType gradientDiff = m_gradientCache[m_index] - m_gradient[m_index];
			
			if ( !isZero(gradientDiff) )
			{
				// compute step of the weight and cache it
				m_stepsCache[m_index] = m_stepsCache[m_index] * m_gradient[m_index] / gradientDiff;

				// cache current gradient
				m_gradientCache[m_index] = m_gradient[m_index];

				// update the weight
				weight += m_stepsCache[m_index];
			}

			// move to the next weight
			++m_index;
		}

		void reset()
		{
			size_t weightsCount = this->m_network.getWeightsCount();
			ConstantInitializer<WeightType>(0)(m_gradientCache, weightsCount);
			ConstantInitializer<WeightType>(-2)(m_stepsCache, weightsCount);
		}

	protected:
		WeightType *m_gradientCache;
		WeightType *m_stepsCache;
		WeightType *m_gradient;
		size_t m_index;

		/** Informs whether the given value is approximately equal to zero. */
		inline bool isZero(const WeightType& weight)
		{
			static const WeightType POS = static_cast<WeightType>( 0.000001f );
			static const WeightType NEG = -POS;
			return (weight > NEG) && (weight < POS);
		}
	};

}

#endif