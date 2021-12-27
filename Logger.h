#ifndef EVENTLOG_H
#define EVENTLOG_H

#include <string>
#include <iostream>
#include <ios>
#include <fstream>
#include <stdexcept>

namespace CML {
  class EventLog {
    public:
      EventLog() { }

      void StartFile_Handler(const std::string& filename);
      void set_log_console_output(bool val);
      bool get_log_console_output();
      std::string get_StartFile();
      void Log_Handler(const std::string& event);
      void Flush_Handler();
      void CloseFile_Handler();

    protected:
      bool log_console_output = false;
      std::string StartFile;
      std::fstream Log;
      static std::streambuf* cout_buf;
      static std::streambuf* cerr_buf;
      int last_flush = 0;
      int flush_period = 5;
  };
}

#endif // EVENTLOG_H
