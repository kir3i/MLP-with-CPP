#include <vector>
#include <iostream>
#include "objective_functions.h"

using namespace std;

// Mean Squared Error
double mean_squared_error(vector<double> y, vector<double> target) {
	if (y.size() != target.size()) {
		cout << "mse error: y.size() != target.size()\n";
		exit(-1);
	}
	double rtn = 0;
	for (int i = 0; i < y.size(); i++)
		rtn += (target[i] - y[i]) * (target[i] - y[i]);
	return rtn / 2;
}