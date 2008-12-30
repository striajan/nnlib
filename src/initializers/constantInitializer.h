#ifndef _CONSTANT_INITIALIZER_H_
#define _CONSTANT_INITIALIZER_H_

#include "initializers/initializer.h"

namespace NNLib
{

	/**
	Initializer that is assigning always the same given value.
	*/
	template <typename T>
	class ConstantInitializer :
		public Initializer<T>
	{
	private:
		typedef Initializer<T> _Initializer;
		
	public:
		typedef typename _Initializer::ValueType ValueType;

		ConstantInitializer(const ValueType& value) :
		m_value(value)
		{ }

		// interface Initializer:
		void operator()(ValueType& initWhat) const
		{
			initWhat = m_value;
		}
		
		void operator()(ValueType initWhat[], size_t len) const
		{
			for (size_t i = 0; i < len; ++i)
				(*this)( initWhat[i] );
		}

	protected:
		const ValueType& m_value;

		ConstantInitializer& operator=(const ConstantInitializer&);
	};

}

#endif