#ifndef _LEARNING_MOMENTUM_H_
#define	_LEARNING_MOMENTUM_H_

namespace NNLib
{

	/**
	Base class for for every learning momentum class.
	*/
	template <typename T>
	class LearningMomentumBase
	{
	public:
		typedef T MomentumType;

	protected:
		static const MomentumType DEF_LEARNING_MOMENTUM;
	};

	template <typename T>
	const typename LearningMomentumBase<T>::MomentumType LearningMomentumBase<T>::DEF_LEARNING_MOMENTUM =
		static_cast<typename LearningMomentumBase<T>::MomentumType>( 0 );

	/**
	This class represents a global learning momentum for a back-propagation algorithm.
	*/
	template <typename T>
	class GlobalLearningMomentum :
		public LearningMomentumBase<T>
	{
	private:
		typedef LearningMomentumBase<T> _LearningMomentumBase;
		
	public:
		typedef typename _LearningMomentumBase::MomentumType MomentumType;

		GlobalLearningMomentum(MomentumType momentum = _LearningMomentumBase::DEF_LEARNING_MOMENTUM) :
		m_momentum(momentum)
		{ }

		inline MomentumType getLearningMomentum() const { return m_momentum; }
		inline void setLearningMomentum(MomentumType momentum) { m_momentum = momentum; }

	protected:
		/** Current value of the momentum. */
		MomentumType m_momentum;
	};
}

#endif