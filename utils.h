#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <functional>

using namespace std;
using namespace boost;

/* Author : Stewart Charles
   Machine Learning Project 
   Utility Class */

namespace ML {

    template <class T>

    /* Data Example Type */
    struct Example {
        T label;
        MlVec<T> data;

        Example(): label(0), data(MlVec<T>()) { }

        Example(T _label, MlVec<T> _data) {
            label = _label;
            data = _data;
        }
    };

    /* Typedef for a Vector of Examples */
    template <class T>
    using Examples = std::vector<Example<T>>;

    /* Calculate the sign of a number (1.0 or -1.0) */
    template <class T>
    T sgn(T val) {
        return (static_cast<T>(0) < val) - (val < static_cast<T>(0)); 
    }

    /* Typedef of Function used to read a line of input data */
    template <class U>
    using ReadFn = std::function<Example<U>(string)>;

    template <class T>
    Examples<T> load_data(string filename, ReadFn<T> rf) {
        Examples<T> examples;
        string line;
        ifstream read(filename);
        if(read.is_open()) 
            while(getline(read,line)) 
                examples.push_back(rf(line));
        else 
            printf("Unable to read file\n");
        return examples;
    }
}