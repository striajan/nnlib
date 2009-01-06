#ifndef _ACCUMULATOR_H_
#define	_ACCUMULATOR_H_

#include <limits>
#include "initializers/constantInitializer.h"

namespace NNLib
{
	
// BASE
	
	template <typename T>
	class AccumulatorBase
	{
	public:
		typedef T ValueType;
	};
	
	
// MEMORY
	
	template <typename T,
		size_t CAPACITY>
	class MemoryAccumulator :
		public AccumulatorBase<T>
	{
	public:
		typedef AccumulatorBase<T> _AccumulatorBase;
		typedef typename _AccumulatorBase::ValueType ValueType;
		
		inline void accum(ValueType val)
		{
			this->next();
			this->add(val);
		}
		
		inline void reset()
		{
			m_size = 0;
			m_pos = getCapacity() - 1;
		}
		
		inline size_t getCapacity() const
		{ return CAPACITY; }
		
	protected:
		ValueType m_memory[CAPACITY];
		size_t m_size;
		size_t m_pos;
		
		inline void next()
		{
			m_pos = (m_pos + 1) % getCapacity();
			if ( m_size < getCapacity() )
				++m_size;
		}
		
		inline void add(ValueType val)
		{ m_memory[m_pos] = val; }
	};
	
	
// LAST VALUE
	
	template <typename T>
	class LastAccumulator :
		public AccumulatorBase<T>
	{	
	public:
		typedef AccumulatorBase<T> _AccumulatorBase;
		typedef typename _AccumulatorBase::ValueType ValueType;
		
		LastAccumulator()
		{ reset(); }
		
		inline void accum(ValueType val)
		{ m_lastValue = val; }
		
		inline ValueType getAccumVal() const
		{ return m_lastValue; }
		
		inline void reset()
		{ m_lastValue = std::numeric_limits<ValueType>::quiet_NaN(); }
		
	protected:
		ValueType m_lastValue;
	};
	
	
// SUM
	
	template <typename T>
	class SumAccumulator :
		public AccumulatorBase<T>
	{	
	public:
		typedef AccumulatorBase<T> _AccumulatorBase;
		typedef typename _AccumulatorBase::ValueType ValueType;
		
		SumAccumulator()
		{ reset(); }
		
		inline void accum(ValueType val)
		{ m_sum += val; }
		
		inline ValueType getAccumVal() const
		{ return m_sum; }
		
		inline void reset()
		{ m_sum = static_cast<ValueType>(0); }
		
	protected:
		ValueType m_sum;
	};
	
	
// MEAN
	
	template <typename T,
		size_t CAPACITY = 0>
	class MeanAccumulator :
		public MemoryAccumulator<T, CAPACITY>
	{
	public:
		typedef MemoryAccumulator<T, CAPACITY> _AccumulatorBase;
		typedef typename _AccumulatorBase::ValueType ValueType;
		
		MeanAccumulator()
		{ reset(); }
		
		inline void accum(ValueType val)
		{
			this->next();
			m_sum -= this->m_memory[this->m_pos];
			m_sum += val;
			this->add(val);
		}
		
		inline ValueType getAccumVal() const
		{ return m_sum / static_cast<ValueType>(this->m_size); }
		
		inline void reset()
		{
			_AccumulatorBase::reset();
			ConstantInitializer<ValueType>(0)(this->m_memory, this->getCapacity());
			m_sum = 0;
		}
		
	protected:
		ValueType m_sum;
	};
	
	template <typename T>
	class MeanAccumulator<T, 0> :
		public AccumulatorBase<T>
	{	
	public:
		typedef AccumulatorBase<T> _AccumulatorBase;
		typedef typename _AccumulatorBase::ValueType ValueType;
		
		MeanAccumulator()
		{ reset(); }
		
		inline void accum(ValueType val)
		{
			m_sum += val;
			++m_size;
		}
		
		inline ValueType getAccumVal() const
		{ return m_sum / static_cast<ValueType>(m_size); }
		
		inline void reset()
		{
			m_sum = static_cast<ValueType>( 0 );
			m_size = 0;
		}
		
	protected:
		ValueType m_sum;
		size_t m_size;
	};
	
	
// MIN
	
	template <typename T,
		size_t CAPACITY = 0>
	class MinAccumulator :
		public MemoryAccumulator<T, CAPACITY>
	{	
	public:
		typedef MemoryAccumulator<T, CAPACITY> _AccumulatorBase;
		typedef typename _AccumulatorBase::ValueType ValueType;
		
		MinAccumulator()
		{ this->reset(); }
		
		ValueType getAccumVal() const
		{
			ValueType min = std::numeric_limits<ValueType>::max();
			for (size_t i = 0; i < this->m_size; ++i)
				if (this->m_memory[i] < min)
					min = this->m_memory[i];
			return min;
		}
	};
	
	template <typename T>
	class MinAccumulator<T, 0> :
		public AccumulatorBase<T>
	{	
	public:
		typedef AccumulatorBase<T> _AccumulatorBase;
		typedef typename _AccumulatorBase::ValueType ValueType;
		
		MinAccumulator() { reset(); }
		
		inline void accum(ValueType val)
		{ if (val < m_min) m_min = val; }
		
		inline ValueType getAccumVal() const
		{ return m_min; }
		
		inline void reset()
		{ m_min = std::numeric_limits<ValueType>::max(); }
		
	protected:
		ValueType m_min;
	};
	

// MAX
	
	template <typename T,
		size_t CAPACITY = 0>
	class MaxAccumulator :
		public MemoryAccumulator<T, CAPACITY>
	{	
	public:
		typedef MemoryAccumulator<T, CAPACITY> _AccumulatorBase;
		typedef typename _AccumulatorBase::ValueType ValueType;
		
		MaxAccumulator()
		{ this->reset(); }
		
		ValueType getAccumVal() const
		{
			ValueType max = std::numeric_limits<ValueType>::min();
			for (size_t i = 0; i < this->m_size; ++i)
				if (this->m_memory[i] > max)
					max = this->m_memory[i];
			return max;
		}
	};

	template <typename T>
	class MaxAccumulator<T, 0> :
		public AccumulatorBase<T>
	{	
	public:
		typedef AccumulatorBase<T> _AccumulatorBase;
		typedef typename _AccumulatorBase::ValueType ValueType;
		
		MaxAccumulator()
		{ reset(); }
		
		inline void accum(ValueType val)
		{ if (val > m_max) m_max = val; }
		
		inline ValueType getAccumVal() const
		{ return m_max; }
		
		inline void reset()
		{ m_max = std::numeric_limits<ValueType>::min(); }
		
	protected:
		ValueType m_max;
	};
	
}

#endif

