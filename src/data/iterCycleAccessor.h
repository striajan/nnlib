#ifndef _CYCLE_ITER_ACCESSOR_H_
#define	_CYCLE_ITER_ACCESSOR_H_

#include <iostream>
#include "data/dataAccessorBase.h"

namespace NNLib
{

	/**
	Access the data in a sequential order in cycles and iterationss.
	*/
	template <typename ContT>
	class IterCycleAccessor :
		public DataAccessorBase<ContT>
	{
	private:
		typedef DataAccessorBase<ContT> _DataAccessorBase;
	
	public:	
		typedef typename _DataAccessorBase::ContainerType ContainerType;
		typedef typename _DataAccessorBase::DataType DataType;
		
		IterCycleAccessor(const ContainerType& container,
		size_t itersCount = 1, size_t cyclesCount = 1) :
		_DataAccessorBase(container),
		m_itersCount(itersCount), m_cyclesCount(cyclesCount)
		{
			begin();
		}
		
		// interface DataAccessor:
		
		inline void begin()
		{
			m_pos = 0;
			m_cycle = m_iter = 1;
			m_end = (this->m_container.size() == 0) ||
				(m_iter > m_itersCount) || (m_cycle > m_cyclesCount);
		}
		
		inline void next()
		{
			if (!m_end) {
				if (m_iter == m_itersCount) {
					if (m_pos == this->m_container.size() - 1) {
						if (m_cycle == m_cyclesCount)
							m_end = true;
						else {
							m_iter = 1;
							m_pos = 0;
							++m_cycle;
						}
					}
					else {
						m_iter = 1;
						++m_pos;
					}
				}
				else
					++m_iter;
			}
		}
		
		inline bool isEnd() const
		{
			return m_end;
		}
		
		inline const DataType& current() const
		{
			return this->m_container[m_pos];
		}
		
	protected:
		/** Current position in a data container. */
		size_t m_pos;
		
		const size_t m_itersCount;
		const size_t m_cyclesCount;
		
		// current iteration and cycle
		size_t m_iter;
		size_t m_cycle;

		/** Informs whether the end was already reached. */
		bool m_end;

		IterCycleAccessor& operator=(const IterCycleAccessor&);
	};

}

#endif