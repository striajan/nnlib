#ifndef _STRINGS_H_
#define _STRINGS_H_

#include <string>
#include <sstream>

namespace NNLib
{
	
	/**
	This class enables converting many arguments to a string.
	*/
	class ToString
	{
	public:
		inline operator std::string() const
		{
			return toString();
		}

		inline std::string toString() const
		{
			return m_ostream.str();
		}

		/** Add argument of any type to the stream. */
		template <class ArgType>
		ToString& operator<<(ArgType const& arg)
		{
			m_ostream << arg;
			return *this;
		}

	private:
		std::ostringstream m_ostream;
	};


	/**
	Macro for a better usage of the class defined above.
	*/
	#define TO_STRING(MSG) ( std::string(ToString() << MSG) )
}

#endif