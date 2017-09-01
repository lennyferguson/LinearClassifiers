cmake_minimum_required(VERSION 3.6)
project(MachineLearningProject)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp nvec.h perceptron.h sgd.h utils.h classifier.h svm.h logreg.h)
add_executable(MachineLearningProject ${SOURCE_FILES})