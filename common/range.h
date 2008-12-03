#ifndef _RANGE_H_
#define _RANGE_H_

namespace NNLib
{

	/**
	Class that represents a range given by its min and max value.
	*/
	template <typename T>
	class Range
	{
	public:
		typedef T ValType;

		Range(const ValType& min, const ValType& max) :
		m_min(min), m_max(max)
		{ }

		inline const ValType& getMin() const { return m_min; }
		inline const ValType& getMax() const { return m_max; }
		inline ValType getRange() const { return m_max - m_min; }

	private:
		ValType m_min;
		ValType m_max;
	};

}

#endif
