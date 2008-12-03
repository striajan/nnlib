#ifndef _NEURON_BASE_H_
#define _NEURON_BASE_H_

#include "../common/exceptions.h"
#include "../common/range.h"
#include "../initializers/initializer.h"

namespace NNLib
{

	/**
	Base neuron implementation that uses a lot of template params.
	*/
	template < typename T,
		template <typename T> class ActT,
		template <typename T> class CombT >
	class NeuronBase
	{
	public:
		typedef T InputType;
		typedef T OutputType;
		typedef T WeightType;
		typedef ActT<T> ActivationFuncType;
		typedef CombT<T> CombinatorType;
		typedef Initializer<T> InitType;

		/** Constructor that takes activation function and combinator as a paramater. */
		NeuronBase(size_t inputsCount,
			const ActivationFuncType *activationFunc,
			const CombinatorType *combinator)
		{
			init(inputsCount, activationFunc, combinator);
			m_activationFuncOwner = m_combinatorOwner = false;
		}

		/** Constructor that creates activation function and combinator. */
		NeuronBase(size_t inputsCount)
		{
			init(inputsCount, new ActivationFuncType(), new CombinatorType());
			m_activationFuncOwner = m_combinatorOwner = true;
		}

		~NeuronBase()
		{
			delete [] m_weights;
			if (m_activationFuncOwner)
				delete m_activationFunc;
			if (m_combinatorOwner)
				delete m_combinator;
		}

		/** Recompute the output of the neuron for the given input and
		store it in the cache. */
		OutputType eval(const InputType input[])
		{
			m_outputCache = m_activationFunc->function( m_combinator->combine(
				input, m_weights, m_inputsCount) );
			return m_outputCache;
		}

		/** Init weights of this neuron with the given initializer. */
		void initWeights(InitType& initializer)
		{
			initializer(m_weights, m_inputsCount);
		}

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
		void init(size_t inputsCount,
			const ActivationFuncType *activationFunc,
			const CombinatorType *combinator)
		{
			m_outputCache = static_cast<OutputType>(0);
			m_inputsCount = inputsCount;
			m_weights = new WeightType[m_inputsCount];
			m_activationFunc = activationFunc;
			m_combinator = combinator;
		}

	protected:
		/** Lastly computed output value of the neuron. */
		OutputType m_outputCache;

		/** Number of inputs of the neuron. The last input should always be 1
		because it's meant to be a special bias input. */
		size_t m_inputsCount;

		/** Input weights of the neuron. */
		WeightType *m_weights;

		/** Activation function for this neuron. */
		const ActivationFuncType *m_activationFunc;
		bool m_activationFuncOwner;

		/** Combinator of the input and weights for this neuron. */
		const CombinatorType *m_combinator;
		bool m_combinatorOwner;

	private:
		NeuronBase& operator=(const NeuronBase&);
	};

}

#endif