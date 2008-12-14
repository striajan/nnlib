#ifndef _TABBED_SIGMOID_FUNC_H_
#define	_TABBED_SIGMOID_FUNC_H_

#include <cmath>
#include <iostream>
#include "activationFunctions/activationFuncBase.h"

namespace NNLib
{
	
	/**
	Sigmoid activation function using a table to store function values.
	                1
	f(x) = ----------------------
	        1 + exp(-lambda * x)
	*/
	template <typename T>
	class TabbedSigmoidFunc :
		public ActivationFuncBase<T>
	{
	private:
		typedef ActivationFuncBase<T> _ActivationFuncBase;
		
	public:
		typedef typename _ActivationFuncBase::ValueType ValueType;
		typedef typename _ActivationFuncBase::ResultType ResultType;
		typedef T ParamType;
		
		TabbedSigmoidFunc()
		{
			if (!s_initialized)
				init();
		}

		// interface ActivationFunc:

		ResultType function(ValueType x) const
		{
			if (x < s_min)
				return static_cast<ResultType>(0);
			if (x > s_max)
				return static_cast<ResultType>(1);
			return s_table[ static_cast<size_t>( (x + s_offset) * s_mult ) ];
		}

		inline ResultType operator()(ValueType x)
		{
			return function(x);
		}

		// interface DerivableActivationFunc:

		ResultType derivation(ValueType x) const
		{
			return valDerivation( function(x) );
		}

		// interface ValDerivableActivationFunc:

		ResultType valDerivation(ResultType y) const
		{
			return static_cast<ResultType>( s_lambda * y * (1 - y) );
		}

		static void init(ParamType lambda = DEF_LAMBDA, ResultType precision = DEF_PRECISION)
		{
			s_lambda = lambda;
			s_precision = precision;
			
			// step in the x to archieve the given precision
			ValueType step = (4 * s_precision) / s_lambda;
			ValueType halfStep = step / 2;
			
			// maximal x to archieve the given precision
			ValueType max = fInv(1 - precision);
			
			ValueType halfRange = ::ceil( (max / step) - halfStep );

			// add one cell to avoid an array overflow
			size_t halfSize = static_cast<size_t>(halfRange) + 1;
			
			s_max = halfRange * step + halfStep;
			s_min = -s_max;
			s_offset = s_max + step;
			
			size_t tableSize = 2 * halfSize + 1;
			
			s_mult = static_cast<ValueType>(tableSize) / (2 * s_offset);
			
			if (s_table != NULL)
				delete [] s_table;
			s_table = new ResultType[tableSize];
			
			s_table[halfSize] = f(0);
			for (size_t i = 1; i <= halfSize; ++i) {
				ValueType x = static_cast<ValueType>(i) * step;
				s_table[halfSize + i] = f(x);
				s_table[halfSize - i] = f(-x);
			}
			
			s_initialized = true;
		}
		
		static void finish()
		{
			s_initialized = false;
			if (s_table != NULL)
				delete [] s_table;
			s_table = NULL;
		}
		
	protected:
		static const ResultType DEF_PRECISION;
		static const ParamType DEF_LAMBDA;
		
		static ResultType s_precision;
		static ParamType s_lambda;
		
		static ValueType s_min;
		static ValueType s_max;
		static ValueType s_mult;
		static ValueType s_offset;
		static ResultType *s_table;
		static bool s_initialized;
		
		
		static ResultType f(ValueType x)
		{
			return static_cast<ResultType>( 1 / (1 + ::exp(-s_lambda * x)) );
		}
		
		static ValueType fInv(ResultType y)
		{
			return static_cast<ValueType>( ::log(1 / y - 1) / -s_lambda );
		}
	};
	
	template <typename T>
	const typename TabbedSigmoidFunc<T>::ResultType TabbedSigmoidFunc<T>::DEF_PRECISION
		= static_cast<typename TabbedSigmoidFunc<T>::ResultType>( 0.001 );
	
	template <typename T>
	const typename TabbedSigmoidFunc<T>::ParamType TabbedSigmoidFunc<T>::DEF_LAMBDA
		= static_cast<typename TabbedSigmoidFunc<T>::ParamType>( 1 );
	
	template <typename T>
	typename TabbedSigmoidFunc<T>::ResultType TabbedSigmoidFunc<T>::s_precision
		= TabbedSigmoidFunc<T>::DEF_PRECISION;
	
	template <typename T>
	typename TabbedSigmoidFunc<T>::ParamType TabbedSigmoidFunc<T>::s_lambda
		= TabbedSigmoidFunc<T>::DEF_LAMBDA;
	
	template <typename T>
	typename TabbedSigmoidFunc<T>::ValueType TabbedSigmoidFunc<T>::s_min;
	
	template <typename T>
	typename TabbedSigmoidFunc<T>::ValueType TabbedSigmoidFunc<T>::s_max;
	
	template <typename T>
	typename TabbedSigmoidFunc<T>::ValueType TabbedSigmoidFunc<T>::s_mult;
	
	template <typename T>
	typename TabbedSigmoidFunc<T>::ValueType TabbedSigmoidFunc<T>::s_offset;
	
	template <typename T>
	typename TabbedSigmoidFunc<T>::ResultType *TabbedSigmoidFunc<T>::s_table
		= NULL;
	
	template <typename T>
	bool TabbedSigmoidFunc<T>::s_initialized = false;
}

#endif