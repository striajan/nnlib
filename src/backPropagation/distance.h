#ifndef _DISTANCE_H_
#define _DISTANCE_H_

#include <cmath>

namespace NNLib
{

	/**
	Base class for every distance functor of two vectors.
	*/
	template <typename T>
	struct DistanceBase
	{
		typedef T InputType;
		typedef T DistType;
	};


	/**
	Computes Manhattan distance of two vectors (also called city-block distance,
	rectilinear distance, L1 distance etc.
	*/
	template <typename T>
	struct ManhattanDistance :
		public DistanceBase<T>
	{
		typedef DistanceBase<T> _DistBase;
		typedef typename _DistBase::InputType InputType;
		typedef typename _DistBase::DistType DistType;

		inline DistType operator()(const InputType x[], const InputType y[], size_t len) const
		{
			return distance(x, y, len);
		}

		inline DistType distance(const InputType x[], const InputType y[], size_t len) const
		{
			DistType dist = 0;
			for (size_t i = 0; i < len; ++i)
				dist += std::abs( x[i] - y[i] );
			return dist;
		}
	};


	/**
	Computes squared value of standard Eucleidian distance.
	*/
	template <typename T>
	struct SquaredEuclideanDistance :
		public DistanceBase<T>
	{
		typedef DistanceBase<T> _DistBase;
		typedef typename _DistBase::InputType InputType;
		typedef typename _DistBase::DistType DistType;

		inline DistType operator()(const InputType x[], const InputType y[], size_t len) const
		{
			return distance(x, y, len);
		}

		inline DistType distance(const InputType x[], const InputType y[], size_t len) const
		{
			DistType dist = 0;
			for (size_t i = 0; i < len; ++i) {
				DistType diff = x[i] - y[i];
				dist += diff * diff;
			}
			return dist;
		}
	};


	/**
	Computes maximal absolute difference of two coordinates.
	*/
	template <typename T>
	struct MaxDistance :
		public DistanceBase<T>
	{
		typedef DistanceBase<T> _DistBase;
		typedef typename _DistBase::InputType InputType;
		typedef typename _DistBase::DistType DistType;

		inline DistType operator()(const InputType x[], const InputType y[], size_t len) const
		{
			return distance(x, y, len);
		}

		inline DistType distance(const InputType x[], const InputType y[], size_t len) const
		{
			DistType max = 0;
			for (size_t i = 0; i < len; ++i) {
				DistType absDiff = std::abs( x[i] - y[i] );
				if (absDiff > max)
					max = absDiff;
			}
			return max;
		}
	};

}

#endif