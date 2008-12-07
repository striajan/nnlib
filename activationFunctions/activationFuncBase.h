#ifndef _ACTIVATION_FUNC_BASE_H_
#define _ACTIVATION_FUNC_BASE_H_

namespace NNLib
{

	/**
	Base template class for every activation function.
	*/
	template <typename T>
	class ActivationFuncBase
	{
	public:
		typedef T ValueType;
		typedef T ResultType;
	};

}

#endif