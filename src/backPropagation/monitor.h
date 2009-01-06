#ifndef _MONITOR_H_
#define _MONITOR_H_

#include <string>
#include <iostream>
#include <iomanip>
#include <vector>

namespace NNLib
{

	/**
	Base class for an every monitor class.
	*/
	class Monitor
	{
	public:
		Monitor(unsigned timer = DEF_TIMER) :
		m_timer(timer), m_counter(0)
		{ }

		inline void operator()()
		{
			if ( ++m_counter == m_timer ) {
				execute();
				m_counter = 0;
			}
		}

		virtual void execute() = 0;
		
		// virtual destructor due to inheritance
		virtual ~Monitor() { }

	protected:
		static const unsigned DEF_TIMER = 1;
		const unsigned m_timer;
		unsigned m_counter;

	private:
		Monitor& operator=(const Monitor&);
	};


	/**
	This monitor does nothing.
	*/
	class EmptyMonitor :
		public Monitor
	{
	public:
		// do nothing
		virtual void execute() { }
	};


	/**
	Combined monitor that holds a list of other monitors and call
	them in the given order.
	*/
	class CombinedMonitor :
		public Monitor
	{
	protected:
		typedef std::vector<Monitor*> MonitorsList;

	public:
		CombinedMonitor(unsigned timer = DEF_TIMER) :
		Monitor(timer)
		{ }

		inline void add(Monitor& monitor)
		{
			m_monitors.push_back(&monitor);
		}

		virtual void execute()
		{
			for (MonitorsList::iterator it = m_monitors.begin(); it != m_monitors.end(); ++it)
				(**it)();
		}

	protected:
		/** List of all the monitors to be called by this monitor. */
		MonitorsList m_monitors;
	};
	
	
	/**
	This monitor prints out the given parameter's value. The parameter has to
	have operator << for printing to ostream overloaded.
	*/
	template <typename ParamT>
	class ParamMonitor :
		public Monitor
	{
	public:
		typedef ParamT ParamType;
		
		ParamMonitor(std::ostream& os, const ParamType& param,
			unsigned timer = DEF_TIMER, const std::string& delim = DEF_DELIM) :
		Monitor(timer), m_ostream(os), m_param(param), m_delim(delim)
		{ }
		
		virtual void execute()
		{
			m_ostream << m_param << std::endl << m_delim;
		}
		
	protected:
		std::ostream& m_ostream;
		const ParamType& m_param;
		const std::string m_delim;
		
		static const char* DEF_DELIM;
		
	private:
		ParamMonitor& operator=(const ParamMonitor&);
	};
	
	template <typename ParamT>
	const char* ParamMonitor<ParamT>::DEF_DELIM = "";
	
	
	/**
	This monitor prints out the given parameter's name and its value. The parameter
	has to have operator << for printing to ostream overloaded.
	*/
	template <typename ParamT>
	class NamedParamMonitor :
		public ParamMonitor<ParamT>
	{
	private:
		typedef ParamMonitor<ParamT> _ParamMonitor;
		
	public:
		typedef typename _ParamMonitor::ParamType ParamType;
		
		NamedParamMonitor(std::ostream& os, const ParamType& param, const std::string& name,
			unsigned timer = _ParamMonitor::DEF_TIMER, const std::string& delim = _ParamMonitor::DEF_DELIM) :
		_ParamMonitor(os, param, timer, delim), m_name(name)
		{ }
		
		virtual void execute()
		{
			this->m_ostream << m_name << std::endl;
			_ParamMonitor::execute();
		}
		
	protected:
		const std::string m_name;
		
	private:
		NamedParamMonitor& operator=(const NamedParamMonitor&);
	};

}

#endif