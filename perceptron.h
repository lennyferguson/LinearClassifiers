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
    class Perceptron : public LinearClassifier<T> {
        public:
        /* Implementation of Perceptron Linear Classifier */
        Perceptron(ML::Examples<T> &examples, int epochs = 1, T gamma = 0.1) {
            
            // Create the Update function used in Stochastic Gradient Descent 
            auto update = [&](MlVec<T> w, ML::Example<T> ex) {
                auto yi = ex.label;
                auto xi = ex.data;
                auto dy = w.dot(xi);
                return (yi * dy <= 0) ? w + yi * gamma * xi : w;
            };
            // Calculate the Weight vector using Stochastic Gradient Descent
            this->weight = ML::SGD<T>(examples,update,epochs);
        }
    };
}