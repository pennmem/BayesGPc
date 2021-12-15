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
    void Log_Handler(const std::string& event);
    void CloseFile_Handler();

    protected:
    bool print_cout = false;
    std::string StartFile;
    std::fstream Log;
    int last_flush = 0;
  };
}

#endif // EVENTLOG_H

