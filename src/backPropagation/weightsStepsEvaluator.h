#ifndef _WEIGHTS_STEPS_EVALUATOR_H_
#define	_WEIGHTS_STEPS_EVALUATOR_H_

#include "feedForward/networkBufferAllocator.h"

namespace NNLib
{
	
	/**
	Base class for every evaluator of weights' steps for back-propagation algorithm.
	*/
	template <typename NetworkT>
	class WeightStepsEvaluatorBase
	{
	public:
		typedef NetworkT NetworkType;
		typedef typename NetworkType::WeightType WeightType;
		typedef typename NetworkType::InputType  InputType;
		typedef typename NetworkType::OutputType OutputType;

		WeightStepsEvaluatorBase(const NetworkType& network) :
		m_network(network)
		{ }

	protected:
		/** Network for which the steps should be evaluated. */
		const NetworkType& m_network;

	private:
		WeightStepsEvaluatorBase& operator=(const WeightStepsEvaluatorBase&);
	};


	/**
	Standard evaluator of weights' steps for back-propagation algorithm using delta
	errors for each neuron.
	*/
	template <typename NetworkT>
	class DeltaWeightsStepsEvaluator :
		public WeightStepsEvaluatorBase<NetworkT>
	{
	private:
		typedef WeightStepsEvaluatorBase<NetworkT> _EvaluatorBase;

	public:
		typedef typename _EvaluatorBase::NetworkType NetworkType;
		typedef typename _EvaluatorBase::WeightType WeightType;
		typedef typename _EvaluatorBase::InputType  InputType;
		typedef typename _EvaluatorBase::OutputType OutputType;
		typedef typename _EvaluatorBase::WeightType DeltaType;

		DeltaWeightsStepsEvaluator(const NetworkType& network) :
		_EvaluatorBase(network)
		{
			m_deltas = createNeuronsBuffer<DeltaType>(this->m_network);
		}

		~DeltaWeightsStepsEvaluator()
		{
			deleteNeuronsBuffer(m_deltas);
		}

		/** Eval errors for the particular neurons. */
		void evalWeightsSteps(const InputType *input, const OutputType *expectedOutput, WeightType ***weightsSteps)
		{
			const size_t layersCount = this->m_network.getLayersCount();

			// eval deltas for all the layers
			evalOutputLayerDeltas(expectedOutput);
			for (size_t layer = 2; layer <= layersCount; ++layer)
				evalHiddenLayerDeltas(layersCount - layer);

			// eval weights steps
			for (size_t layer = layersCount - 1; layer > 0; --layer)
				evalLayerWeightsSteps(layer, this->m_network[layer-1].getOutputCache(), weightsSteps[layer]);
			evalLayerWeightsSteps(0, input, weightsSteps[0]);
		}

	protected:
		/** Deltas for the neurons. */
		DeltaType **m_deltas;

		/** Eval deltas for the output layer and for the given expected output. This method
		supposes that output of the output layer is cached in it. */
		void evalOutputLayerDeltas(const OutputType expectedOutput[])
		{
			const size_t layer = this->m_network.getLayersCount() - 1;
			const size_t neuronsCount = this->m_network[layer].getNeuronsCount();
			const OutputType *realOutput = this->m_network[layer].getOutputCache();
			DeltaType *deltas = m_deltas[layer];

			// difference between the expected and real output
			for (size_t i = 0; i < neuronsCount; ++i)
				deltas[i] = ( expectedOutput[i] - realOutput[i] ) *
					this->m_network[layer][i].getActivationFunc().valDerivation( realOutput[i] );
		}

		/** Eval deltas for the given hidden layer. This method supposes that output
		of the layer is cached in it. */
		void evalHiddenLayerDeltas(size_t layer)
		{
			const size_t nextLayer = layer + 1;
			const size_t neuronsCount = this->m_network[layer].getNeuronsCount();
			const size_t nextNeuronsCount = this->m_network[nextLayer].getNeuronsCount();
			const OutputType *output = this->m_network[layer].getOutputCache();
			DeltaType *deltas = m_deltas[layer];
			const DeltaType *nextDeltas = m_deltas[nextLayer];

			for (size_t i = 0; i < neuronsCount; ++i)
			{
				// compute weighted sum of deltas from the next layer
				DeltaType nextDeltasSum = 0;
				for (size_t j = 0; j < nextNeuronsCount; ++j)
					nextDeltasSum += this->m_network[nextLayer][j][i] * nextDeltas[j];

				// compute delta for each neuron from this layer
				deltas[i] = nextDeltasSum *
					this->m_network[layer][i].getActivationFunc().valDerivation( output[i] );
			}
		}

		/** Eval weight steps for the given layer. */
		void evalLayerWeightsSteps(size_t layer, const InputType input[], WeightType **weightsStepsLayer)
		{
			const size_t inputsCount = this->m_network[layer].getInputsCount();
			const size_t neuronsCount = this->m_network[layer].getNeuronsCount();
			const DeltaType *deltas = m_deltas[layer];
			WeightType *weightsSteps = *weightsStepsLayer;

			size_t k = 0;
			for (size_t j = 0; j < neuronsCount; ++j)
				for (size_t i = 0; i < inputsCount; ++i)
					weightsSteps[k++] = deltas[j] * input[i];
		}

	private:
		DeltaWeightsStepsEvaluator& operator=(const DeltaWeightsStepsEvaluator&);
	};
	
}

#endif