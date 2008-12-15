#ifndef _IN_OUT_DATA_H_
#define _IN_OUT_DATA_H_

#include <vector>
#include "common/utils.h"
#include "common/exceptions.h"

namespace NNLib
{
	
	template <typename T>
	class InOutData;
	
	
	template <typename T>
	class InOutPair
	{
	public:
		typedef T InputType;
		typedef T OutputType;
		typedef const InputType*  InputVector;
		typedef const OutputType* OutputVector;
		
		InOutPair(InputVector input, OutputVector output)
		{
			m_input = input;
			m_output = output;
		}
		
		inline InputVector getInput() { return m_input; }
		inline InputVector getOutput() { return m_output; }
		
	private:
		friend class InOutData< InOutPair<T> >;
		
		void deleteData()
		{
			delete [] m_input;
			delete [] m_output;
		}
		
		/** Vector of input data. */
		InputVector m_input;
		
		/** Vector of output data. */
		OutputVector m_output;
	};


	/**
	Data for supervised learning and/or testing consist of input and expected
	output pairs.
	*/
	template <typename T>
	class InOutData
	{
	public:
		typedef T Pair;
		typedef Pair value_type;   // for a compatibility with standard containers
		typedef typename Pair::InputType  InputType;
		typedef typename Pair::OutputType OutputType;
		typedef typename Pair::InputVector  InputVector;
		typedef typename Pair::OutputVector OutputVector;
		typedef std::vector<Pair> DataContainer;

		InOutData(size_t inputLen, size_t outputLen) :
		m_inputLen(inputLen), m_outputLen(outputLen)
		{ }
		
		~InOutData()
		{
			for (typename DataContainer::iterator it = m_data.begin(); it != m_data.end(); ++it)
				it->deleteData();
			m_data.clear();
		}

		/** Add input and an expected output for it. */
		void add(InputVector in, OutputVector out)
		{
			InputVector inCopy = createAndCopyArray(in, m_inputLen);
			OutputVector outCopy = createAndCopyArray(out, m_outputLen);
			m_data.push_back( Pair(inCopy, outCopy) );
		}
		
		/** Get a pair of data with range checking (slower but safer). */
		inline const Pair& getPair(size_t index) const
		{
			if ( index < 0 || index >= size() )
				throw IndexOutOfArray( index, size() );
			return (*this)[index];
		}
		
		/** Get a pair of data without range checking. */
		inline const Pair& operator[](size_t index) const
		{
			return m_data[index];
		}
		
		inline size_t size() const { return m_data.size(); }
		
		inline size_t getInputLen() const { return m_inputLen; }
		inline size_t getOutputLen() const { return m_outputLen; }

	private:
		// input and output vector lengths
		const size_t m_inputLen;
		const size_t m_outputLen;

		/** Container of pairs of input and expected output vectors. */
		DataContainer m_data;

		InOutData& operator=(const InOutData&);
		InOutData(const InOutData&);
	};

}

#endif