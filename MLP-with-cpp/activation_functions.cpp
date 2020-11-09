#include <cmath>
#include "activation_functions.h"

// Hard Limiting (기준값: 0)
double hard_limiting(double x) {
	if (x <= 0)
		return 0;
	else
		return 1;
}

// Sigmoid 
double Sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

// Sigmoid 도함수
double SigmoidPrime(double x) {
	return Sigmoid(x) * (1 - Sigmoid(x));
}
