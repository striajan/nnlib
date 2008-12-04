#ifndef _IN_OUT_DATA_H_
#define _IN_OUT_DATA_H_

#include <vector>
#include <utility>
#include "../common/utils.h"

namespace NNLib
{

	/**
	A bunch of training or testing data.
	*/
	template <typename T>
	class InOutData
	{
	public:
		typedef T InputType;
		typedef T OutputType;
		typedef const InputType*  InputVector;
		typedef const OutputType* OutputVector;
		typedef std::pair<InputVector, OutputVector> DataType;
		typedef std::vector<DataType> DataContainer;

		InOutData(size_t inputLen, size_t outputLen) :
		m_inputLen(inputLen), m_outputLen(outputLen)
		{ }

		/** Add <input, expected output> data pair. */
		void add(DataType data)
		{
			m_data.push_back(data);
		}

		/** Add input and an expected output for it. */
		void add(InputVector in, OutputVector out)
		{
			m_data.push_back( DataType(in, out) );
		}

		/** Add copy of the input and an expected output vector. */
		void addCopy(InputVector in, OutputVector out)
		{
			InputVector inCopy = createAndCopyArray<T>(in, m_inputLen);
			OutputVector outCopy = createAndCopyArray<T>(out, m_outputLen);
			add(inCopy, outCopy);
		}

	private:
		// input and output vector lengths
		const size_t m_inputLen;
		const size_t m_outputLen;

		/** Container of pairs of input and expected output vectors. */
		DataContainer m_data;

		InOutData& operator=(const InOutData&);
	};

}

#endif