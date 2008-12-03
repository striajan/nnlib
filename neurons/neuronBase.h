#ifndef _NEURON_BASE_H_
#define _NEURON_BASE_H_

#include "../common/exceptions.h"
#include "../activationFunctions/sigmoidFunc.h"
#include "../combinators/dotProduct.h"

namespace NNLib
{

	/**
	Base neuron implementation that uses a lot of template params.
	*/
	template < typename T,
		template <typename T> class ActFunc = SigmoidFunc,
		template <typename T> class Comb = DotProductCombinator >
	class NeuronBase
	{
	public:
		typedef T InputType;
		typedef T OutputType;
		typedef T WeightType;
		typedef ActFunc<T> ActivationFunc;
		typedef Comb<T> Combinator;

		/** Parametric constructor. */
		NeuronBase(size_t inputsCount,
			const ActivationFunc& activationFunc,
			const Combinator& combinator) :
		m_inputsCount(inputsCount),
		m_activationFunc(activationFunc), m_combinator(combinator)
		{
			m_weights = new WeightType[m_inputsCount];
		}

		~NeuronBase()
		{
			delete [] m_weights;
		}

		/** Recompute the output of the neuron for the given input and
		store it in the cache. */
		OutputType eval(const InputType input[])
		{
			m_outputCache = m_activationFunc( m_combinator(input, m_weights, m_inputsCount) );
			return m_outputCache;
		}

	public:
		/** Get weight with range checking. */
		inline const WeightType& getWeight(size_t index) const
		{
			if (index < 0 || index >= m_inputsCount)
				throw IndexOutOfArray(index, m_inputsCount);
			return m_weights[index];
		}

		inline const WeightType& getBias() const { return m_weights[m_inputsCount-1]; }

		/** Methods without range checkin (but faster). */
		inline const WeightType& operator[](size_t index) const { return m_weights[index]; }
		inline WeightType& operator[](size_t index) { return m_weights[index]; }

		inline size_t getInputsCount() const { return m_inputsCount; }
		inline OutputType getOutputCache() const { return m_outputCache; }

	protected:
		/** Number of inputs of the neuron. The last input should always be 1
		because it's meant to be a special bias input. */
		size_t m_inputsCount;

		/** Lastly computed output value of the neuron. */
		OutputType m_outputCache;

		/** Input weights of the neuron. */
		WeightType *m_weights;

		/** Activation function for this neuron. */
		const ActivationFunc& m_activationFunc;

		/** Combinator of the input and weights for this neuron. */
		const Combinator& m_combinator;
	};

}

#endif