#ifndef _INITIALIZER_H_
#define _INITIALIZER_H_

#include "initializers/initializerBase.h"

namespace NNLib
{

	/**
	Abstract initializer of objects of the given type.
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
		virtual void operator()(ValueType initWhat[], size_t len) const
		{
			for (size_t i = 0; i < len; ++i)
				(*this)( initWhat[i] );
		}
		
		/** Init a single value. */
		virtual void operator()(ValueType&) const = 0;

		virtual ~Initializer() = 0;
	};
	
	template <typename T>
	Initializer<T>::~Initializer()
	{ }

}

#endif