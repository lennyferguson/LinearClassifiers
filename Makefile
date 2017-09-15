machine_learning : main.cpp classifier.h logreg.h nvec.h \
					perceptron.h sgd.h svm.h utils.h
					g++ -std=c++11 -O3 -march=native -o machine_learning main.cpp
clean :
	rm machine_learning