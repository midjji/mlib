#pragma once
#ifndef STRING_HELPERS_H
#define STRING_HELPERS_H
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "mlib/utils/vector.h"
typedef unsigned int uint;

template<class T> std::string str(T t){
    std::stringstream ss;
    ss<<t;
    return ss.str();
}


namespace mlib
{
// string helpers
template<class T> std::string toStr(T t, unsigned int res=0){
    std::stringstream ss;
    if (res == 0)
        ss << t;
    else
        ss << std::setprecision(res) << t;
    return ss.str();
}
std::string pad(std::string in, uint len, char key=' ');


std::vector<std::string> split_lines(std::string s, char delim='\n');


double str2double(const char* str);
int str2int(const char* str);
double str2double(const std::string& str);
int str2int(const std::string& str);

std::string toLowerCase(std::string in);

template<class T> std::string toZstring(T i, uint z=5)
{
    std::stringstream ss;
    ss<<std::setfill('0') << std::setw(z)<<i;
    return ss.str();
}



/* the opposite is
 * std::istream is; is >> t; if(is) return t; std::optional?*/



bool equal(std::string a, std::string b);


template <typename T>
std::ostream& operator<<(std::ostream& os,
                         const std::vector<T>& v)
{
    os << "Size: " << v.size() << "\n [";
    for (const T& t : v) os << t << ", ";
    os << "]";
    return os;
}

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& os,
                                const std::map<T1, T2>& m)
{
    if (m.empty()) {
        os << "[]";
        return os;
    }
    os << "[";
    for (auto it = m.begin(); it != m.end(); ++it) {
        if (it != m.begin()) os << ", ";
        os << "(" << it->first << ": " << it->second << ")\n";
    }
    os << "]";
    return os;
}

/// sprintf with std::string output
std::string s_printf(const char *fmt, ...);






template<class T>
int getStringWidth(const T& obj){
    std::stringstream ss;
    ss<<obj;
    return (int)ss.str().size();
}

template<class T>
std::vector<int> getStringWidths(const std::vector<T>& objs){
    std::vector<int> ws;ws.reserve(objs.size());
    for(const T& obj:objs)
        ws.push_back(getStringWidth(obj));
    return ws;
}


template<class T> std::vector<std::string>  toStrVec(std::vector<T> elements){
    std::vector<std::string> strs;strs.reserve(elements.size());
    for(T e:elements){
        std::stringstream ss;
        ss<<e;
        strs.push_back(ss.str());
    }
    return strs;
}

template<class T> std::vector<std::vector<std::string>>
toStrMat(std::vector<std::vector<T>> rows){
    std::vector<std::vector<std::string>> strrows;strrows.reserve(rows.size());
    for(std::vector<T> e:rows)
        strrows.push_back(toStrVec(e));
    // verify all rows have the same length by adding empty
    uint values=0;
    for(const std::vector<std::string>& row:strrows)
        values=std::max(values,(uint)row.size());

    for(uint i=0;i<strrows.size();++i)
        strrows[i].resize(values," ? ");
    //for(auto e:strrows){        for(auto s:e)       std::cout<<s<<", ";    std::cout<<std::endl;    }
    return strrows;
}

template<class T> std::vector<T> push_first(std::vector<T> ts,T v){
    std::vector<T> tmps;tmps.reserve(ts.size()+1);tmps.push_back(v);
    for(auto& t:ts)    tmps.push_back(t);
    return tmps;
}

template<class T>
/**
 * @brief displayTable
 * @param headers
 * @param rows
 * @return a string containing a table convenient for display purposes
 */
std::string displayTable(std::vector<std::string> headers,
                         std::vector<std::vector<T>> valrows,
                         std::vector<std::string> rownames=std::vector<std::string>(),
                         std::string title="Table title"){


    // first convert it to a string matrix, filling missing data with " ? "
    std::vector<std::vector<std::string>> rows=toStrMat(valrows);

    // if both headers and rownames is 0 they should not be included!
    if(rownames.size()!=0 && rows.size()>0){

        // then if rownames is non zero size make it fullsize
        rownames.resize(rows.size()," ? ");

        // add them to the beginning of the matrix
        for(uint i=0;i<rows.size();++i)
            rows[i]=push_first(rows[i],rownames[i]);

        // then if headers is too short first add table to the start
        if(headers.size()<rows[0].size())
            headers=push_first(headers,std::string("Table"));

        while(headers.size()<rows[0].size())
            headers.push_back(" ? ");
    }

    // now headers and rows are of equal size, make it to a big matrix
    if(headers.size()>0)
        rows=push_first(rows,headers);

    // now precompute the size of the columns






    int min_header_width=7;


    std::vector<int> widths;
    assert(rows.size()>0);
    if(rows.size()>0)widths.resize(rows[0].size(),min_header_width);

    // header widths fixed, now for row widths
    for(auto row:rows){
        for(uint e=0;e<row.size();++e){
            widths[e]=std::max(getStringWidth(row[e]),widths[e]);
        }
    }

    // all widths are computed!



    std::stringstream ss0,ss;
    int rowwidth=0;

    // per row
    for(uint row=0;row<rows.size();++row)
    {

        ss<<" | ";
        for(uint element=0;element<rows[row].size();++element)
        {
            ss <<" "<< std::left << std::setw(widths[element]) << std::setfill(' ') << rows[row][element]<<" |";
        }
        if(row==0)
            rowwidth=(int)ss.str().size();
        ss<<"\n";
    }
    // add roof, data and floor

    std::string spaces; spaces.resize((rowwidth-title.size())/2,' ');

    ss0<<" "<<spaces<<title<<"\n";
    ss0<<" "<<std::left << std::setw(std::max(rowwidth-1,0)) << std::setfill('-')<<" "<<"\n";
    ss0<<ss.str();
    ss0<<" "<<std::left << std::setw(std::max(rowwidth-1,0)) << std::setfill('-')<<" "<<"\n";


    return ss0.str();
}
/**
 * @brief display - summarize a std::vectors content
 * @param xs
 * @param showfull - include the entire vector
 * @return a string showing common usefull statistics for the vector of doubles
 */
template<class T> std::string display(const std::vector<T>& xs,bool showfull=false)
{




    if (xs.size() == 0) return "[]";
    std::stringstream ss;
    T avg = mean(xs);
    T medi = median(xs);
    T mi = min(xs);
    T ma = max(xs);
    if(!showfull){
        std::vector<std::string> headers={"mean","median","min","max","samples"};
        std::vector<std::vector<T>> data={{avg,medi,mi,ma,(T)xs.size()}};

        return displayTable(headers,data,std::vector<std::string>(),"displaying vector");

    }
    ss << "mean median min max\n";
    ss << avg << " " << medi << " " << mi << " " << ma;

    if (xs.size() < 5 || showfull) {
        ss << "\n";
        for(uint i=0;i<xs.size();++i){
            ss<<round(xs[i]*100)/100.0<<" ";
            if(i % 20 ==19)
                ss<<"\n";
        }
    }
    return ss.str();
}
template<class T> std::string display(const std::vector<std::vector<T>>& xs,std::vector<std::string> rownames){
    std::vector<std::string> headers={"mean","median","min","max"};
    std::vector<std::vector<T>> data;
    for(std::vector<T> x:xs){
        if(x.size()>0){
            T avg = mean(x);
            T medi = median(x);
            T mi = min(x);
            T ma = max(x);
            data.push_back({avg,medi,mi,ma});
        }
        else{
            data.push_back({});
        }
    }
    return displayTable(headers,data,rownames);
}
/*
template<class T>
void clear_and_reset_screenbuffer(){
    cout << "\033[2J\033[1;1H"; cout.flush();
}
*/






}  // end namespace mlib



#endif  // STRING_HELPERS_H
