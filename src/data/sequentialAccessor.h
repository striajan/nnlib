#ifndef _SEQUENTIAL_ACCESSOR_H_
#define	_SEQUENTIAL_ACCESSOR_H_

#include "data/dataAccessorBase.h"

namespace NNLib
{

	/**
	Access the data in a sequential order (as they're stored in the container).
	*/
	template <typename ContT>
	class SequentialAccessor :
		public DataAccessorBase<ContT>
	{
	private:
		typedef DataAccessorBase<ContT> _DataAccessorBase;
	
	public:	
		typedef typename _DataAccessorBase::ContainerType ContainerType;
		typedef typename _DataAccessorBase::DataType DataType;
		
		SequentialAccessor(const ContainerType& container) :
		_DataAccessorBase(container)
		{
			begin();
		}
		
		// interface DataAccessor:
		
		inline void begin()
		{
			m_counter = 0;
		}
		
		inline void next()
		{
			if ( !isEnd() )
				++m_counter;
		}
		
		inline bool isEnd() const
		{
			return ( m_counter == this->m_container.size() );
		}
		
		inline const DataType& current() const
		{
			return this->m_container[m_counter];
		}
		
	protected:
		/** Current position in a data container. */
		size_t m_counter;

		SequentialAccessor& operator=(const SequentialAccessor&);
	};

}

#endif