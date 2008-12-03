#ifndef _EXCEPTIONS_H_
#define _EXCEPTIONS_H_

#include <stdexcept>
#include "strings.h"

namespace NNLib
{

	/**
	Exception that should be thrown when an index exceeds bounds of an array.
	*/
	class IndexOutOfArray :
		public std::out_of_range
	{
	public:
		IndexOutOfArray(size_t index, size_t size) :
		m_index(index), m_size(size), std::out_of_range( createMsg(index, size) )
		{ }

		inline size_t getIndex() const { return m_index; }
		inline size_t getArraySize() const { return m_size; }

		static std::string createMsg(size_t index, size_t size)
		{
			return TO_STRING("IndexOutOfArray Exception: index " << index <<
				" is out of an array's bounds 0.." << size - 1);
		}

	protected:
		size_t m_index;
		size_t m_size;
	};

}

#endif