#ifndef _FEED_FORWARD_LAYER_
#define _FEED_FORWARD_LAYER_

#include "common/exceptions.h"

namespace NNLib
{

	template <typename NeuronT>
	class FeedForwardLayer
	{
	public:
		typedef NeuronT NeuronType;
		typedef NeuronType *NeuronPtr;
		typedef typename NeuronType::InputType InputType;
		typedef typename NeuronType::OutputType OutputType;

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
			return m_outputsCache[i];
		}

		/** Get neuron with range checking and not NULL checking. */
		const NeuronType& getNeuron(size_t index) const
		{
			if (index < 0 || index >= m_neuronsCount)
				throw IndexOutOfArray(index, m_neuronsCount);
			if (m_weights[index] == NULL)
				throw NullPointerException("pointer to NULL couldn't be dereferenced");
			return (*this)[index];
		}

		// methods without range checkin and not NULL checking
		inline const NeuronType& operator[](size_t index) const { return m_neurons[index]; }
		inline NeuronType& operator[](size_t index) { return m_neurons[index]; }

		inline size_t getNeuronsCount() const { return m_neuronsCount; }
		inline size_t getInputsCount() const { return m_inputsCount; }
		inline size_t getOutputsCount() const { return m_neuronsCount + 1; }

		inline const FeedForwardLayer* getPrevLayer() const { return m_prev; }
		inline void setPrevLayer(const FeedForwardLayer *prev) { m_prev = prev; }

		inline const FeedForwardLayer* getNextLayer() const { return m_next; }
		inline void setNextLayer(const FeedForwardLayer *next) { m_next = next; }

		inline bool isInputLayer() const { return m_prev == NULL; }
		inline bool isOutputLayer() const { return m_next == NULL; }
		inline bool isHiddenLayer() const { return !isInputLayer() && !isOutputLayer(); }

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
			for (size_t i = 0; i < m_neuronsCount; ++i)
				delete m_neurons[i];
			delete [] m_neurons;
			delete [] m_outputsCache;
		}
	};

}

#endif