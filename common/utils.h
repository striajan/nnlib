#ifndef _UTILS_H_
#define _UTILS_H_

namespace NNLib
{

	template <typename T>
	void copyArray(const T src[], T dest[], size_t len)
	{
		for (size_t i = 0; i < len; ++i)
			dest[i] = src[i];
	}

	template <typename T>
	T* createAndCopyArray(const T src[], size_t len)
	{
		T *dest = new T[len];
		copyArray<T>(src, dest, len);
		return dest;
	}

}

#endif