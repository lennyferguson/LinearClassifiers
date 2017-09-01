#pragma once

#include "utils.h"

namespace ML {

    /* Author : Stewart Charles
       Machine Learning Project 
       LinearClassifier super class */
    template <class T>
    class LinearClassifier {
        public:
        LinearClassifier() {};
        
        /* */
        virtual T classify(const MlVec<T> &ex) {
            return ML::sgn(weight.dot(ex));
        }
        
        protected:
        MlVec<T> weight;
    };
}