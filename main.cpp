#include "nvec.h"
#include "utils.h"
#include "perceptron.h"
#include "svm.h"
#include "logreg.h"
#include <iostream>
#include <cassert>
#include <string>
#include <map>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

using namespace std;
using namespace ML;
using namespace boost;

constexpr size_t get_dim(const unsigned int d) {
  return 20 * d + 1;
}

using namespace ML;
/* Testing Functions */
int main(int argc, char** argv) {
  string train_file = "_datasets/746Data.txt";
  string test_file = "_datasets/1625Data.txt";
  if(argc == 3) {
    train_file = argv[1];
    test_file  = argv[2];
  }

  /* */
  const size_t DIM = get_dim(8);

  /* Character map for Data that maps an 'Amino Acid' to a particular binary label.
     For use with HIV1_PR datasets. */
  map<char,int> char_map = 
    {{'A',0} , {'R',1}  , {'N',2}  , {'D',3}  , {'C',4}  , {'Q',5},
    {'E',6}  , {'G',7}  , {'H',8}  , {'I',9}  , {'L',10} , {'K',11} , {'M',12},
    {'F',13} , {'P',14} , {'S',15} , {'T',16} , {'W',17} , {'Y',18} , {'V',19}};

  /* Write function to parse line of the input data file, 
     since data can be in many different formats. */
  auto parse_line = [&] (string line) {
    MlVec<double> v(DIM);
    v[0] = 1.0;
    const int offset = 20;
    vector<string> split_row;
    split(split_row,line,is_any_of(","));
    auto left = split_row[0];
    auto right = split_row[1];
    double label = atof(right.c_str());
    for(int i = 0, index = 1; i < left.length(); i++, index += offset)
      v[index + char_map[left[i]]] = 1.0;
    return Example<double>(label,v);
  };

  /* Load the Testing and Training data. */
  auto training_data = load_data<double>(train_file, parse_line);
  auto test_data = load_data<double>(test_file, parse_line);

  /* Measure the performance of the Classifier */
  double per_correct = 0.0;
  double svm_correct = 0.0;
  double log_correct = 0.0;

  ML::SVM<double>        s(training_data,10,0.01,100.0);
  ML::Perceptron<double> p(training_data,10,0.01);
  ML::LogReg<double>     l(training_data,50,0.01,50);

  for(auto &ex : test_data) {
    if(ex.label * s.classify(ex.data) > 0)
      svm_correct += 1.0;
    if(ex.label * p.classify(ex.data) > 0)
      per_correct += 1.0;
    if(ex.label * l.classify(ex.data) > 0)
      log_correct += 1.0;
  }
  printf("Perceptron Percent Correct: %f\n", per_correct / test_data.size());
  printf("SVM Percent Correct:        %f\n", svm_correct / test_data.size());
  printf("Log Reg Percent Correct:    %f\n", log_correct / test_data.size());
}

/* MlVec tests */
void test() {
  // Use Variadic Constructor!
  MlVecD a(1.0,1.0,1.0,1.0);

  // Or use Initialize List!
  MlVecD b = { 2.0, 3.0, 4.0, 5.0 };

  assert(a == a && b == b);
  assert(a != b && b != a);

  auto c = a + b;
  assert(c == MlVecD(3.0,4.0,5.0,6.0));

  c = b - a;
  assert(c == MlVecD(1.0,2.0,3.0,4.0));

  c = b * b;
  assert(c == MlVecD(4.0,9.0,16.0,25.0));

  assert(a * 2 == 2 * a && a * 2 ==  MlVecD(2.0,2.0,2.0,2.0));

  c = b / b;
  assert(c == MlVecD(1.0,1.0,1.0,1.0));

  c = a / 2;
  assert(a / 2 == a * (1.0 / 2.0) && a / 2 == MlVecD(0.5,0.5,0.5,0.5));
  assert(a / 2 == a / MlVecD(2.0,2.0,2.0,2.0));

  assert(a * b == b * a && a * b == MlVecD(2.0,3.0,4.0,5.0));

  c = a * b + b * a;
  assert(c.dot(c) == 216.0);

  c = a.dot(b);
  assert(c == 14.0);

  cout << "Tests Passed" << endl;
}
