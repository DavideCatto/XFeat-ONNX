#pragma once

#include <iostream>

// LOG
#ifdef _DEBUG
#define DEBUG_MSG(str) do { std::cout << str << std::endl; } while( false )
#else
#define DEBUG_MSG(str) do { } while ( false )
#endif


struct Configuration
{
    std::string xfeatPath;
    std::string matcherPath;
    bool isDense = true;
    bool isEndtoEnd = false;
    std::string device;
    bool show = true;
};