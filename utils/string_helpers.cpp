#include "mlib/utils/string_helpers.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>

using namespace std;

namespace mlib
{
std::string pad(std::string in, uint len, char key){
    while(in.size()<len)
        in.push_back(key);
    return in;
}

// string helpers

double str2double(char *str)
{
    std::istringstream ss(str);
    double d;
    ss >> d;
    return d;
}

int str2int(char *str)
{
    std::istringstream ss(str);
    int d;
    ss >> d;
    return d;
}

std::string toLowerCase(std::string in)
{
    std::string data = in;
    std::transform(data.begin(), data.end(), data.begin(), ::tolower);
    return data;
}


std::string ensure_dir(std::string path){
    if(path.back()=='/')
        return path;
    return path+"/";
}




bool equal(std::string a, std::string b)
{
    return (a.compare(b) == 0);
}






double roundTo(double d, int digits)
{
    std::stringstream ss;
    ss << std::setprecision(digits) << d;
    double tmp;
    ss >> tmp;
    return tmp;
}

std::string s_printf(const char *fmt, ...)
{ // replace with std::format
    string s;

    va_list args;

    va_start(args, fmt);
    int n = vsnprintf(nullptr, 0, fmt, args) + 1;
    va_end(args);

    s.resize(n, 'x');

    va_start(args, fmt);
    vsnprintf(const_cast<char*>(s.data()), n, fmt, args);
    va_end(args);

    return s;
}
std::vector<std::string> split_lines(std::string s, char delim)
{

    std::vector<std::string> lines;
    if(s.size()==0)
        return lines;




    size_t last = 0, next = 0;
    while ((next = s.find(delim, last)) != std::string::npos) {

        lines.push_back(s.substr(last, next-last));
        last = next + 1;
    }

    lines.push_back(s.substr(last));


    return lines;
}






}  // end namespace mlib
