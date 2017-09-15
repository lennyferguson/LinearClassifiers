#ifndef __SGD_H__
#define __SGD_H__

#include <functional>
#include <algorithm>
#include "nvec.h"
#include "util.h"
#include <algorithm>
#include <cstdlib>

using namespace std;

/* Author : Stewart Charles
   Machine Learning Project
   Stochastic Gradient Descent */

namespace ML {

    template <class U>
    using Func = std::function<MlVec<U>(MlVec<U>,ML::Example<U>)>;

    /* Stochastic Gradient Descent algorithm */
    template <class T>
    MlVec<T> SGD(ML::Examples<T> examples, Func<T> fn, int epochs = 1) {
        const size_t DIM = examples[0].data.len();
        MlVec<T> w(DIM);
        for(int e = 0; e < epochs; e++) {
            random_shuffle(examples.begin(), examples.end());
            for(auto ex : examples)
                w = fn(w,ex);
        }
        return w;
    }
}
#endif