#include "Logger.h"

namespace CML {
  void EventLog::StartFile_Handler(const std::string& filename) {
    Log.open(filename, std::ios::out);
    last_flush = 0;
  }

  void EventLog::Log_Handler(const std::string& event) {
    if (Log) {
      Log << event;
      if (last_flush > 5) {
        Log.flush();
        last_flush = 0;
      }
      else { last_flush++; }
    }
    else {
        throw std::runtime_error("Logger file stream not open.");
    }
  }

  void EventLog::CloseFile_Handler() {
    Log.close();
  }
}
