#pragma once

#include "utils.h"

namespace ML {

    template <class U>
    using Func = std::function<MlVec<U>(MlVec<U>,ML::Example<U>)>;
  
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

	static MlVec<T> SGD(ML::Examples<T> examples, Func<T> fn, int epochs = 1) {
	  const size_t DIM = examples[0].data.len();
	  MlVec<T> w(DIM);
	  for(int e = 0; e < epochs; e++) {
            random_shuffle(examples.begin(), examples.end());
            for(auto ex : examples)
	      w = fn(w,ex);
	  }
	  return w;
	}
    };
}
