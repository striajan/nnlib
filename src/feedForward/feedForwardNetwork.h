#ifndef _FEED_FORWARD_NETWORK_
#define _FEED_FORWARD_NETWORK_

#include <vector>
#include "common/exceptions.h"
#include "common/utils.h"

namespace NNLib
{

	/**
	This class represents a whole feed-forward neural network.
	*/
	template <typename LayerT>
	class FeedForwardNetwork
	{
	public:
		typedef LayerT LayerType;
		typedef LayerType *LayerPtr;
		typedef typename LayerType::NeuronType NeuronType;
		typedef typename LayerType::InputType InputType;
		typedef typename LayerType::OutputType OutputType;
		typedef typename NeuronType::WeightType WeightType;
		typedef typename std::vector<LayerPtr> LayersList;
		typedef std::vector<size_t> LayersSizes;

		FeedForwardNetwork()
		{ }

		FeedForwardNetwork(size_t inputsCount, const LayersSizes& sizes)
		{
			create(inputsCount, sizes);
		}

		~FeedForwardNetwork()
		{
			destroy();
		}

		const OutputType* eval(const InputType inputs[])
		{
			typename LayersList::iterator end = m_layers.end();
			for (typename LayersList::iterator it = m_layers.begin(); it != end; ++it)
				inputs = (*it)->eval(inputs);
			return inputs;
		}

		void initWeights(const Initializer<WeightType>& initializer)
		{
			typename LayersList::iterator end = m_layers.end();
			for (typename LayersList::iterator it = m_layers.begin(); it != end; ++it)
				(*it)->initWeights(initializer);
		}

		void pushLayer(LayerType *layer)
		{
			if (layer == NULL)
				throw NullPointerException("pointer to a layer to be added is NULL");
			
			// bind the last layer with the newly added
			LayerType *last = m_layers.empty() ? NULL : m_layers.back();
			if (last != NULL)
				last->setNextLayer(layer);
			layer->setPrevLayer(last);

			m_layers.push_back(layer);
		}

		const LayerType& getLayer(size_t index) const
		{
			if (index < 0 || index >= m_layers.size())
				throw IndexOutOfArray(index, m_layers.size());
			if (m_layers[index] == NULL)
				throw NullPointerException("pointer to NULL couldn't be dereferenced");
			return (*this)[index];
		}

		inline const LayerType& operator[](size_t index) const { return *m_layers[index]; }
		inline LayerType& operator[](size_t index) { return *m_layers[index]; }

		inline size_t getLayersCount() const { return m_layers.size(); }

		inline size_t getInputsCount() const { return m_layers.front()->getInputsCount(); }
		inline size_t getOutputsCount() const { return m_layers.back()->getNeuronsCount(); }

		size_t getNeuronsCount() const
		{
			size_t neuronsSum = 0;
			for (size_t layer = 0; layer < getLayersCount(); ++layer)
				neuronsSum += (*this)[layer].getNeuronsCount();
			return neuronsSum;
		}

		size_t getWeightsCount() const
		{
			size_t weightsSum = 0;
			for (size_t layer = 0; layer < getLayersCount(); ++layer)
				weightsSum += (*this)[layer].getWeightsCount();
			return weightsSum;
		}

		inline const OutputType* getOutputCache() const
		{
			return m_layers.back()->getOutputCache();
		}

	protected:
		/** List of all the layers of the network. */
		LayersList m_layers;

	protected:
		void create(size_t inputsCount, const LayersSizes& sizes)
		{
			// create the first layer with the given count of inputs
			LayerPtr layer = new LayerType(sizes[0], inputsCount);
			pushLayer(layer);

			// create next layers - each one has number of inputs that is equal to the number
			// of outputs of the previous layer
			for (size_t i = 1; i < sizes.size(); ++i) {
				layer = new LayerType( sizes[i], layer->getOutputsCount() );
				pushLayer(layer);
			}
		}

		void destroy()
		{
			deleteRange(m_layers.begin(), m_layers.end());
			m_layers.clear();
		}
	};

}

#endif