#ifndef _ACTIVATION_FUNC_H_
#define _ACTIVATION_FUNC_H_

namespace NNLib
{

	/**
	Interface for a common activation function of a neuron.
	*/
	template <typename T, typename R = T>
	class ActivationFunc
	{
	public:
		typedef T ValueType;
		typedef R ResultType;

		/** Evaluate the function for the given 'x'. */
		virtual R function(T x) const = 0;

		/** Evaluate the function for the given 'x'. */
		inline R operator()(T x) const
		{
			return function(x);
		}

		virtual ~ActivationFunc() = 0 { }
	};

}

#endif