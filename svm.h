#pragma once

#include <vector>
#include "nvec.h"
#include "sgd.h"
#include "utils.h"
#include "classifier.h"

/* Author : Stewart Charles
   Machine Learning Project */

namespace ML {

    template <class T>
    class SVM : public LinearClassifier<T> {
        public:
        /* Implementation of Perceptron Linear Classifier */
        SVM(ML::Examples<T> &examples, int epochs = 1, T gamma = 0.1, T C = 1.0) {
            long long t = 1;
            // Create the Update function used in Stochastic Gradient Descent 
            auto update = [&](MlVec<T> w, ML::Example<T> ex) {
                T gamma_t = (gamma / (1.0 + gamma * static_cast<T>(t++)));
                auto yi = ex.label;
                auto xi = ex.data;
                auto dy = w.dot(xi);
                auto wt = w * (1.0 - gamma_t);
                if (yi * dy <= 1.0)
                    wt = wt + gamma_t * C * yi * xi;
                return wt;
            };
            // Calculate the Weight vector using Stochastic Gradient Descent
            this->weight = ML::SGD<T>(examples,update,epochs);
        }
    };
}