#include "Logger.h"

namespace CML {
  std::streambuf* EventLog::cout_buf = nullptr;
  std::streambuf* EventLog::cerr_buf = nullptr;

  void EventLog::StartFile_Handler(const std::string& filename) {
    StartFile = filename;
    Log.open(StartFile, std::ios::out);
    if (log_console_output) {
      if (cout_buf) { throw std::runtime_error("cout buffer already reassigned."); }
      if (cerr_buf) { throw std::runtime_error("cerr buffer already reassigned."); }

      cout_buf = std::cout.rdbuf();
      std::cout.rdbuf(Log.rdbuf());
      cerr_buf = std::cerr.rdbuf();
      std::cerr.rdbuf(Log.rdbuf());
    }

    last_flush = 0;
  }

  void EventLog::Log_Handler(const std::string& event) {
    if (!log_console_output) { std::cout << "LOG: " << event; }
    // TODO still would like to programmatically redirect cout and cerr to log file... but then we'd lose granularity
    // TODO would also like log filtering functionality, I guess this could be implemented with grep or something similar
    //      so long as log messages are properly labeled, as I think they are in RC logging
    if (Log) {
      Log << event;
      // TODO are we sure we want buffering given we'll lose error messages in crashes?
      // TODO could we have separate error message logging without buffering at the very least?
      if (last_flush > flush_period) {
        Log.flush();
        last_flush = 0;
      }
      else { last_flush++; }
    }
    else {
        throw std::runtime_error("Logger file stream not open.");
    }
  }
  
  void EventLog::set_log_console_output(bool val) {
    log_console_output = val;
  }

  bool EventLog::get_log_console_output() {
    return log_console_output;
  }

  std::string EventLog::get_StartFile() {
    return StartFile;
  }

  void EventLog::Flush_Handler() {
    Log.flush();
    last_flush = 0;
  }

  void EventLog::CloseFile_Handler() {
    if (log_console_output && cout_buf) {
      std::cout.rdbuf(cout_buf);
      cout_buf = nullptr;
      std::cerr.rdbuf(cerr_buf);
      cerr_buf = nullptr;
    }

    Log.flush();
    Log.close();
  }
}
