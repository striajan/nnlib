#ifndef _INITIALIZER_H_
#define _INITIALIZER_H_

#include "initializerBase.h"

namespace NNLib
{

	/**
	Abstract initializer of an array of objects of the given type.
	*/
	template <typename T>
	class Initializer :
		public InitializerBase<T>
	{
	private:
		typedef InitializerBase<T> _InitializerBase;
		
	public:
		typedef typename _InitializerBase::ValueType ValueType;
		
		/** Init array of the given length. */
		virtual void operator()(ValueType[], size_t) const = 0;

		virtual ~Initializer() = 0;
	};
	
	template <typename T>
	Initializer<T>::~Initializer()
	{ }

}

#endif