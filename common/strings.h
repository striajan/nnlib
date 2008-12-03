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
		operator std::string() const
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

	#define TO_STRING(MSG) ( std::string(ToString() << MSG) )
}

#endif