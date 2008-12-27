#ifndef _NEURON_BASE_H_
#define _NEURON_BASE_H_

#include "common/exceptions.h"
#include "initializers/initializer.h"

namespace NNLib
{

	/**
	Base neuron implementation that uses a lot of template params.
	*/
	template < typename T,
		template <typename T> class ActivationFuncT,
		template <typename T> class CombinatorT >
	class NeuronBase
	{
	public:
		typedef T InputType;
		typedef T OutputType;
		typedef T WeightType;
		typedef ActivationFuncT<T> ActivationFuncType;
		typedef CombinatorT<T> CombinatorType;

		/** Basic constructor. */
		NeuronBase(size_t inputsCount) :
		m_inputsCount(inputsCount)
		{
			m_weights = new WeightType[m_inputsCount];
		}

		~NeuronBase()
		{
			delete [] m_weights;
		}

		/** Recompute the output of the neuron for the given input and
		store it in the cache. */
		inline OutputType evalAndCache(const InputType input[])
		{
			return ( m_outputCache = eval );
		}

		/** Recompute the output of the neuron for the given input. */
		inline OutputType eval(const InputType input[])
		{
			return m_activationFunc( m_combinator(input, m_weights, m_inputsCount) );
		}

		/** Init weights of this neuron with the given initializer. */
		void initWeights(Initializer<WeightType>& initializer)
		{
			initializer(m_weights, m_inputsCount);
		}

		/** Get weight with range checking. */
		const WeightType& getWeight(size_t index) const
		{
			if (index < 0 || index >= m_inputsCount)
				throw IndexOutOfArray(index, m_inputsCount);
			return m_weights[index];
		}

		WeightType* getWeights() const { return m_weights; }

		inline const WeightType& getBias() const { return m_weights[m_inputsCount-1]; }

		// methods without range checking
		inline const WeightType& operator[](size_t index) const { return m_weights[index]; }
		inline WeightType& operator[](size_t index) { return m_weights[index]; }

		inline size_t getInputsCount() const { return m_inputsCount; }
		inline const OutputType& getOutputCache() const { return m_outputCache; }

		inline const ActivationFuncType& getActivationFunc() const { return m_activationFunc; }
		inline const CombinatorType& getCombinator() const { return m_combinator; }

	protected:
		/** Lastly computed output value of the neuron. */
		OutputType m_outputCache;

		/** Number of inputs of the neuron. The last input should always be 1
		because it's meant to be a special bias input. */
		size_t m_inputsCount;

		/** Input weights of the neuron. */
		WeightType *m_weights;

		/** Activation function for this neuron. */
		ActivationFuncType m_activationFunc;

		/** Combinator of the input and weights for this neuron. */
		CombinatorType m_combinator;
	};

}

#endif