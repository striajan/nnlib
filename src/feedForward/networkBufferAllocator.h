#ifndef _NETWORK_BUFFER_ALLOCATOR_
#define _NETWORK_BUFFER_ALLOCATOR_

#include "common/utils.h"

namespace NNLib
{

	/** Create buffers of the same structure as is the structure of the
	network's neurons. */
	template <typename T, typename NetworkT>
	T** createNeuronsBuffer(const NetworkT& network)
	{
		const size_t layersCount = network.getLayersCount();
		const size_t neuronsCount = network.getNeuronsCount();

		T** buff = new T*[layersCount];
		buff[0] = new T[neuronsCount];

		// pointer from layers array to neurons array
		size_t neuronsSum = 0;
		for (size_t layer = 0; layer < layersCount; ++layer) {
			buff[layer] = buff[0] + neuronsSum;
			neuronsSum += network[layer].getNeuronsCount();
		}

		return buff;
	}

	/** Delete buffer of values of the same structure as is the structure
	of the network's neurons. */
	template <typename T>
	void deleteNeuronsBuffer(T** buff)
	{
		delete [] *buff;
		delete [] buff;
	}

	/** Create buffers of the same structure as is the structure of the
	network's weights. */
	template <typename T, typename NetworkT>
	T*** createWeightsBuffer(const NetworkT& network)
	{
		const size_t layersCount  = network.getLayersCount();
		const size_t neuronsCount = network.getNeuronsCount();
		const size_t weightsCount = network.getWeightsCount();

		T*** buff = new T**[layersCount];
		buff[0] = new T*[neuronsCount];
		buff[0][0] = new T[weightsCount];

		size_t neuronsSum = 0, weightsSum = 0;
		for (size_t layer = 0; layer < layersCount; ++layer)
		{
			// pointer from layers array to neurons array
			buff[layer] = buff[0] + neuronsSum;
			neuronsSum += network[layer].getNeuronsCount();

			// pointers from neurons array to weights array
			for (size_t neuron = 0; neuron < network[layer].getNeuronsCount(); ++neuron) {
				buff[layer][neuron] = buff[0][0] + weightsSum;
				weightsSum += network[layer].getInputsCount();
			}
		}

		return buff;
	}

	/** Delete buffer of values of the same structure as is the structure
	of the network's weights. */
	template <typename T>
	void deleteWeightsBuffer(T*** buff)
	{
		delete [] **buff;
		delete [] *buff;
		delete [] buff;
	}

}

#endif