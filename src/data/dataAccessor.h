#ifndef _DATA_ACCESSOR_H_
#define	_DATA_ACCESSOR_H_

#include "data/dataAccessorBase.h"

namespace NNLib
{

	/**
	Interface for a common data accessor.
	*/
	template <typename ContT>
	class DataAccessor :
		public DataAccessorBase<ContT>
	{
	private:
		typedef DataAccessorBase<ContT> _DataAccessorBase;
	
	public:	
		typedef typename _DataAccessorBase::ContainerType ContainerType;
		typedef typename _DataAccessorBase::DataType DataType;
		
		DataAccessor(const ContainerType& container) :
		_DataAccessorBase(container)
		{ }
		
		virtual ~DataAccessor() = 0;
		
		/** Reset accessor (start from a begin). */
		virtual void begin() = 0;
		
		/** Move to the next member of the container. */
		virtual void next() = 0;
		
		/** Informs whether the end of the data has been reached. */
		virtual bool isEnd() = 0;
		
		/** Get data on the current position. */
		virtual const DataType& current() const = 0;
	};
	
	template <typename ContT>
	DataAccessor<ContT>::~DataAccessor()
	{ }

}

#endif

