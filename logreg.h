#pragma once

#include <vector>
#include "nvec.h"
#include "sgd.h"
#include "utils.h"
#include "classifier.h"

/* Author : Stewart Charles
   Machine Learning Project 
   Implementation of Logistic Regression classifier */

namespace ML {

    template <class T>
    class LogReg : public LinearClassifier<T> {
        public:
        /* Implementation of Perceptron Linear Classifier */
        LogReg(ML::Examples<T> &examples, int epochs = 1, T gamma = 0.1, T sigma = 1) {
            T div = 1.0 / (sigma * sigma);
            // Create the Update function used in Stochastic Gradient Descent 
            auto update = [&](MlVec<T> w, ML::Example<T> ex) {
                auto yi  = ex.label;
                auto xi  = ex.data;
                auto reg = div * w.dot(w);
                T s = log(1.0 + exp(-w.dot(xi))) + reg;
                return w - ((ex.label < 0 ? 0 : 1) - s) * ex.data;
            };
            // Calculate the Weight vector using Stochastic Gradient Descent
            this->weight = ML::SGD<T>(examples,update,epochs);
        }

        T classify(const MlVec<T> &ex) {
        return log(1.0 + exp(-this->weight.dot(ex))) > 0.5 ? 1.0 : -1.0;
    }
    };
}