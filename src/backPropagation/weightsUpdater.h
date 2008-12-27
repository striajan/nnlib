#ifndef _WEIGHTS_UPDATER_H_
#define _WEIGHTS_UPDATER_H_

#include "feedForward/networkBufferAllocator.h"
#include "backPropagation/learningRate.h"
#include "backPropagation/learningMomentum.h"

namespace NNLib
{

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


	/**
	Standard version of weights updater with a global learning rate and momentum.
	*/
	template <typename NetworkT>
	class StandardUpdater :
		public WeightsUpdaterBase<NetworkT>,
		public GlobalLearningRate<typename NetworkT::WeightType>,
		public GlobalLearningMomentum<typename NetworkT::WeightType>
	{
	protected:
		typedef WeightsUpdaterBase<NetworkT> _UpdaterBase;

	public:
		typedef typename _UpdaterBase::NetworkType NetworkType;
		typedef typename _UpdaterBase::WeightType WeightType;

		StandardUpdater(NetworkType& network) :
		_UpdaterBase(network)
		{
			m_weightsCache = createWeightsBuffer<WeightType>(this->m_network);
			for (size_t i = 0; i < this->m_network.getWeightsCount(); ++i)
				(**m_weightsCache)[i] = static_cast<WeightType>(0);
			m_weightsLinCache = m_weightsCache[0][0];
		}

		~StandardUpdater()
		{
			deleteWeightsBuffer(m_weightsCache);
		}

		void updateWeights(WeightType ***weightsSteps)
		{
			const WeightType *steps = weightsSteps[0][0];
			const size_t layersCount = this->m_network.getLayersCount();

			size_t k = 0;
			for (size_t layer = 0; layer < layersCount; ++layer)
			{
				const size_t neuronsCount = this->m_network[layer].getNeuronsCount();
				const size_t inputsCount = this->m_network[layer].getInputsCount();

				for (size_t neuron = 0; neuron < neuronsCount; ++neuron)
				{
					for (size_t input = 0; input < inputsCount; ++input)
					{
						// update the cached weight and remeber the old value
						WeightType w = m_weightsLinCache[k];
						m_weightsLinCache[k] = this->m_network[layer][neuron][input];

						// update the weight
						this->m_network[layer][neuron][input] +=
							this->getLearningRate() * steps[k] +
							this->getMomentum() * (m_weightsLinCache[k] - w);

						// move to the next weight
						++k;
					}
				}
			}
		}

	protected:
		/** Cache holding old weights. */
		WeightType ***m_weightsCache;
		WeightType *m_weightsLinCache;
	};

}

#endif