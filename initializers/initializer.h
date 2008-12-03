#ifndef _INITIALIZER_H_
#define _INITIALIZER_H_

namespace NNLib
{

	/**
	Abstract initializer of an array of objects of the given type.
	*/
	template <typename T>
	class Initializer
	{
	public:
		typedef T ValType;

		/** Init array of the given length. */
		virtual void operator()(ValType[], size_t) const = 0;

		virtual ~Initializer() = 0 { }
	};

}

#endif