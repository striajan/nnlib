#ifndef _RANDOM_ACCESSOR_H_
#define	_RANDOM_ACCESSOR_H_

#include "data/dataAccessorBase.h"
#include "common/random.h"

namespace NNLib
{

	/**
	Access the date in a random order.
	*/
	template <typename ContT>
	class RandomAccessor :
		public DataAccessorBase<ContT>
	{
	private:
		typedef DataAccessorBase<ContT> _DataAccessorBase;
		typedef RandomUniform<size_t> _Random;
	
	public:	
		typedef typename _DataAccessorBase::ContainerType ContainerType;
		typedef typename _DataAccessorBase::DataType DataType;
		
		RandomAccessor(const ContainerType& container, size_t stepsCount = 0) :
		_DataAccessorBase(container),
		m_random( _Random::RangeType(0, container.size()-1) ),
		m_stepsCount(stepsCount)
		{
			m_random.reset();
			begin();
		}
		
		// interface DataAccessor:
		
		inline void begin()
		{
			m_step = 0;
			next();
		}
		
		inline void next()
		{
			if ( !isEnd() ) {
				++m_step;
				m_counter = m_random.next();
			}
		}
		
		inline bool isEnd() const
		{
			return ( m_step == m_stepsCount );
		}
		
		inline const DataType& current() const
		{
			return this->m_container[m_counter];
		}
		
	protected:
		/** Random numbers generator. */
		_Random m_random;
		
		/** Total number of steps. */
		const size_t m_stepsCount;
		
		/** Number of the current step. */
		size_t m_step;
		
		/** Current position in a data container. */
		size_t m_counter;

		RandomAccessor& operator=(const RandomAccessor&);
	};

}

#endif