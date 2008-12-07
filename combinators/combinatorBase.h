#ifndef _COMBINATOR_BASE_H_
#define _COMBINATOR_BASE_H_

namespace NNLib
{

	/**
	Base template class for every combinator.
	*/
	template <typename T>
	class CombinatorBase
	{
	public:
		typedef T InputType;
		typedef T OutputType;
	};

}

#endif