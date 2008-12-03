#ifndef _ACTIVATION_FUNC_H_
#define _ACTIVATION_FUNC_H_

namespace NNLib
{

	/**
	Interface for a common activation function of a neuron.
	*/
	template <typename T>
	class ActivationFunc
	{
	public:
		typedef T ValueType;
		typedef T ResultType;

		/** Evaluate the function for the given 'x'. */
		virtual ResultType function(ValueType x) const = 0;

		/** Evaluate the function for the given 'x'. */
		inline ResultType operator()(ValueType x) const
		{
			return function(x);
		}

		virtual ~ActivationFunc() = 0 { }
	};

}

#endif