#ifndef _DOT_PRODUCT_SSE_H_
#define _DOT_PRODUCT_SSE_H_

#include <xmmintrin.h>
#include "combinators/combinatorBase.h"

namespace NNLib
{

	/**
	Functor that computes dot product of two arrays using SSE instructions.
	*/
	template <typename T>
	class DotProductSSE;
	
	
	/**
	Specialization for a float type.
	*/
	template <>
	class DotProductSSE<float> :
		public CombinatorBase<float>
	{
	public:
		
		// interface Combinator:

		OutputType combine(const InputType x[], const InputType y[], size_t len) const
		{
			float z = 0.0f;
			float sum = 0.0f;

			// 16 bytes aligned array - compiler specific
			#if defined _MSC_VER
				__declspec(align(16)) float ftmp[4];
			#elif defined __GNUC__
				float ftmp[4] __attribute__ ((aligned (16))) = {0.0f, 0.0f, 0.0f, 0.0f};
			#else
				#error unknown compiler (SSE instructions not supported)
			#endif

			__m128 mres;

			// compute the part of the vector which length is dividable by 4
			if ( (len / 4) != 0 )
			{
				mres = _mm_load_ss(&z);
				for (size_t i = 0; i < len; i += 4) {
					mres = _mm_add_ps( mres,
						_mm_mul_ps( _mm_loadu_ps(&x[i]), _mm_loadu_ps(&y[i]) ) );
				}

				// mres = a, b, c, d
				__m128 mv1 = _mm_movelh_ps(mres, mres);   // a, b, a, b
				__m128 mv2 = _mm_movehl_ps(mres, mres);   // c, d, c, d
				mres = _mm_add_ps(mv1, mv2);              // res[0], res[1]

				_mm_store_ps(ftmp, mres);

				sum = ftmp[0] + ftmp[1];
			}

			// compute the remaining part of the vector
			if ( (len % 4) != 0 )
			{
				for (size_t i = len - len % 4; i < len; ++i)
				sum += x[i] * y[i];
			}

			return sum;
		}

		inline OutputType operator()(const InputType x[], const InputType y[], size_t len) const
		{
			return combine(x, y, len);
		}
	};

}

#endif

