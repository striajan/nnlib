#ifndef _FEED_FORWARD_LAYER_
#define _FEED_FORWARD_LAYER_

#include "common/exceptions.h"
#include "common/utils.h"
#include "initializers/initializer.h"

namespace NNLib
{

	/**
	This class represents a sigle layer of a feed-forward neural network.
	*/
	template <typename NeuronT>
	class FeedForwardLayer
	{
	public:
		typedef NeuronT NeuronType;
		typedef NeuronType *NeuronPtr;
		typedef typename NeuronType::InputType InputType;
		typedef typename NeuronType::OutputType OutputType;
		typedef typename NeuronType::WeightType WeightType;

		FeedForwardLayer(size_t neuronsCount, size_t inputsCount)
		{
			create(neuronsCount, inputsCount);
		}

		~FeedForwardLayer()
		{
			destroy();
		}

		const OutputType* eval(const InputType input[])
		{
			for (size_t i = 0; i < m_neuronsCount; ++i)
				m_outputsCache[i] = m_neurons[i]->eval(input);
			return m_outputsCache;
		}

		void initWeights(const Initializer<WeightType>& initializer)
		{
			for (size_t i = 0; i < m_neuronsCount; ++i)
				initializer(m_neurons[i]->getWeights(), m_inputsCount);
		}

		/** Get neuron with range checking and not NULL checking. */
		const NeuronType& getNeuron(size_t index) const
		{
			if (index < 0 || index >= m_neuronsCount)
				throw IndexOutOfArray(index, m_neuronsCount);
			if (m_neurons[index] == NULL)
				throw NullPointerException("pointer to NULL couldn't be dereferenced");
			return (*this)[index];
		}

		/** Perform the given function on every input weight of this layer. */
		template <typename Function>
		inline void forEachWeightForward(Function& func)
		{
			for (size_t neuron = 0; neuron < m_neuronsCount; ++neuron)
				for (size_t input = 0; input < m_inputsCount; ++input)
					func( (*this)[neuron][input] );
		}

		/** Perform the given function on every neuron of this layer. */
		template <typename Function>
		inline void forEachNeuronForward(Function& func)
		{
			for (size_t neuron = 0; neuron < m_neuronsCount; ++neuron)
				func( (*this)[neuron] );
		}

		// methods without range checkin and not NULL checking
		inline const NeuronType& operator[](size_t index) const { return *m_neurons[index]; }
		inline NeuronType& operator[](size_t index) { return *m_neurons[index]; }

		inline size_t getNeuronsCount() const { return m_neuronsCount; }
		inline size_t getInputsCount() const { return m_inputsCount; }
		inline size_t getOutputsCount() const { return m_neuronsCount + 1; }
		inline size_t getWeightsCount() const { return getNeuronsCount() * getInputsCount(); }

		inline const FeedForwardLayer* getPrevLayer() const { return m_prev; }

		inline const FeedForwardLayer* setPrevLayer(const FeedForwardLayer *prev)
		{
			const FeedForwardLayer *old = m_prev;
			if ( (prev != NULL) && (prev->getOutputsCount() != getInputsCount()) )
				throw NonConsistentLayersException( prev->getOutputsCount(), getInputsCount() );
			m_prev = prev;
			return old;
		}

		inline const FeedForwardLayer* getNextLayer() const { return m_next; }

		inline const FeedForwardLayer* setNextLayer(const FeedForwardLayer *next)
		{
			const FeedForwardLayer *old = m_next;
			if ( (next != NULL) && (next->getInputsCount() != getOutputsCount()) )
				throw NonConsistentLayersException( getOutputsCount(), next->getInputsCount() );
			m_next = next;
			return old;
		}

		inline bool isInputLayer() const { return m_prev == NULL; }
		inline bool isOutputLayer() const { return m_next == NULL; }
		inline bool isHiddenLayer() const { return !isInputLayer() && !isOutputLayer(); }

		inline const OutputType* getOutputCache() const { return m_outputsCache; }

	protected:
		/** Count of neurons in this layer. */
		size_t m_neuronsCount;

		/** Count of inputs for each neuron in this layer. */
		size_t m_inputsCount;

		/** List of all neurons in this layer. */
		NeuronPtr *m_neurons;

		/** Input for this layer. */
		OutputType *m_outputsCache;

		/** Pointer to the previous layer (NULL for the first layer). */
		const FeedForwardLayer *m_prev;

		/** Pointer to the previous layer (NULL for the last layer). */
		const FeedForwardLayer *m_next;

	protected:
		void create(size_t neuronsCount, size_t inputsCount)
		{
			m_neuronsCount = neuronsCount;
			m_inputsCount = inputsCount;

			m_neurons = new NeuronPtr[m_neuronsCount];
			for (size_t i = 0; i < m_neuronsCount; ++i)
				m_neurons[i] = new NeuronType(inputsCount);

			m_outputsCache = new OutputType[m_neuronsCount + 1];
			m_outputsCache[m_neuronsCount] = 1;  // the last output is always 1
			// and it repesents a bias input for the next layer

			m_prev = m_next = NULL;
		}

		void destroy()
		{
			deleteRange(m_neurons, m_neurons + m_neuronsCount);
			delete [] m_neurons;
			delete [] m_outputsCache;
		}
	};

}

#endif