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

	protected:
		size_t m_index;
		size_t m_size;

		static std::string createMsg(size_t index, size_t size)
		{
			return TO_STRING("IndexOutOfArray Exception: index " << index <<
				" is out of an array's bounds 0.." << size - 1);
		}
	};


	/**
	Unexpected null pointer exception.
	*/
	class NullPointerException :
		public std::runtime_error
	{
	public:
		NullPointerException(const std::string& msg) :
		std::runtime_error( TO_STRING("NullPointerException: " << msg) )
		{ }
	};

	/**
	Layers don't match on each other.
	*/
	class NonConsistentLayersException :
		public std::runtime_error
	{
	public:
		NonConsistentLayersException(size_t outputs, size_t inputs) :
		m_outputs(outputs), m_inputs(inputs),
		std::runtime_error( createMsg(outputs, inputs) )
		{ }

		inline size_t getOutputs() const { return m_outputs; }
		inline size_t getInputs() const { return m_inputs; }

	protected:
		size_t m_outputs;
		size_t m_inputs;

		static std::string createMsg(size_t outputs, size_t inputs)
		{
			return TO_STRING("NonConsistentLayersException: layer with " << outputs <<
				" outputs can't be connected with a layer with " << inputs << " inputs");
		}
	};

}

#endif