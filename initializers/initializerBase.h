#ifndef _INITIALIZER_BASE_H_
#define _INITIALIZER_BASE_H_

namespace NNLib
{

	/**
	Base template class for every initializer.
	*/
	template <typename T>
	class InitializerBase
	{
	public:
		typedef T ValueType;
	};

}

#endif