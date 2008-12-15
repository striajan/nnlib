#ifndef _DATA_ACCESSOR_BASE_H_
#define	_DATA_ACCESSOR_BASE_H_

namespace NNLib
{

	/**
	Base template class for every data accessor.
	*/
	template <typename ContT>
	class DataAccessorBase
	{
	public:
		typedef ContT ContainerType;
		typedef typename ContainerType::value_type DataType;
		
		DataAccessorBase(const ContainerType& container) :
		m_container(container)
		{ }
		
	protected:
		/** Data to be accessed stored in a container. */
		const ContainerType& m_container;

		DataAccessorBase& operator=(const DataAccessorBase&);
	};

}

#endif

