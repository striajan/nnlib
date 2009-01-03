#ifndef _MATHEMATICS_H_
#define _MATHEMATICS_H_

namespace NNLib
{

	/** Computes signum of the given value. */
	template <typename T>
	inline T sgn(const T& val)
	{
		static const T ZERO_VAL = static_cast<T>(0);
		static const T NEG_VAL  = static_cast<T>(-1);
		static const T POS_VAL  = static_cast<T>(1);
		
		if (val > ZERO_VAL) return POS_VAL;
		else if (val < ZERO_VAL) return NEG_VAL;
		else return ZERO_VAL;
	}

	template <typename T>
	inline const T& max(const T& x, const T& y)
	{
		return ( (x > y) ? x : y );
	}

	template <typename T>
	inline const T& min(const T& x, const T& y)
	{
		return ( (x < y) ? x : y );
	}

}

#endif