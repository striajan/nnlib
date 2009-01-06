#ifndef _CYCLE_ITER_ACCESSOR_H_
#define	_CYCLE_ITER_ACCESSOR_H_

#include <ostream>
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
		m_itersCount(itersCount), m_pattsCount( container.size() ),
		m_cyclesCount(cyclesCount), m_cycleLen( itersCount * container.size() ),
		m_totalLen( itersCount * container.size() * cyclesCount )
		{
			begin();
		}
		
		// interface DataAccessor:
		
		inline void begin()
		{
			m_pos = m_globalPos = 0;
		}
		
		inline void next()
		{
			++m_globalPos;
			m_pos = (m_globalPos % m_cycleLen) / m_itersCount;
		}
		
		inline bool isEnd() const
		{
			return (m_pattsCount == 0) ||
				( (m_globalPos >= m_totalLen) && (m_totalLen > 0) );
		}
		
		inline const DataType& current() const
		{
			return this->m_container[m_pos];
		}
		
		inline size_t getIter() const { return (m_globalPos % m_itersCount) + 1; }
		inline size_t getItersCount() const { return m_itersCount; }
		inline size_t getPatt() const { return m_pos + 1; }
		inline size_t getPattsCount() const { return m_pattsCount; }
		inline size_t getCycle() const { return (m_globalPos / m_cycleLen) + 1; }
		inline size_t getCyclesCount() const { return m_cyclesCount; }
		inline size_t getProgress() const { return m_globalPos + 1; }
		inline size_t getTotalLen() const { return m_totalLen; }
		
	protected:
		const size_t m_itersCount;
		const size_t m_pattsCount;
		const size_t m_cyclesCount;
		const size_t m_cycleLen;
		const size_t m_totalLen;
		size_t m_globalPos;
		size_t m_pos;

	private:
		IterCycleAccessor& operator=(const IterCycleAccessor&);
	};


	/** Print informations about this iter-pattern-cycle accessor
	to the given output stream. */
	template <typename ContT>
	std::ostream& operator<<(std::ostream& os, const IterCycleAccessor<ContT>& access)
	{
		const bool inf = (access.getCyclesCount() == 0);
		os << "iter=" << access.getIter() << "/" << access.getItersCount() <<
			" pattern=" << access.getPatt() << "/" << access.getPattsCount() <<
			" cycle=" << access.getCycle() << "/";
		inf ? (os << "inf") : ( os << access.getCyclesCount() );
		os << " total=" << access.getProgress() << "/";
		inf ? (os << "inf") : ( os << access.getTotalLen() );
		return os;
	}

}

#endif