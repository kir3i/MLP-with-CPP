#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include "activation_functions.h"
#include "Perceptron.h"

using namespace std;


// Perceptron ������
// n: �Է� ����, lr: learning rate
Perceptron::Perceptron(int n, double lr) {
	this->input_dim = n;				// �Է� ���� �ʱ�ȭ
	this->weight_random_initialize();	// weight �ʱ�ȭ
	this->learning_rate = lr;			// learning_rate �ʱ�ȭ
}

// weight�� ��ȯ�Ѵ�.
vector<double> Perceptron::get_weights() {
	return weights;
}

// weight�� �ֿܼ� ����Ѵ�.
void Perceptron::print_weights() {
	cout << "weights: ";
	for (const double &w : weights) {
		cout << w << " ";
	}
	cout << "\n";
}

// weight�� ���� ������ �������� ����Ѵ�.
// 2���� �Է¿��� ��ȿ
void Perceptron::print_linear_function() {
	// �Է� ���� üũ
	if (input_dim != 2)
		return;

	// ������ ������ ���
	cout << "������ ������: x2 = ";
	cout << (weights[0] / weights[1] > 0 ? "-" : "") << abs(weights[0] / weights[1]) << " * x1";
	cout << (weights[2] / weights[1] > 0 ? " - " : " + ") << abs(weights[2] / weights[1]) << "\n";
}

// ������ �������� ���Ϸ� �����Ѵ�.
// 2���� �Է¿��� ��ȿ
// outFile: ������ �������� ����� ����
void Perceptron::write_linear_function(ofstream &outFile) {
	// �Է� ���� üũ
	if (input_dim != 2)
		return;

	// ������ ������ ����
	outFile << (-1 * weights[0] / weights[1]) << " " << (-1 * weights[2] / weights[1]) << "\n";
}

// forward ����
// x: input���� ������ vector
// ��ȯ��: forward ���� ���
double Perceptron::forward(const vector<double> &x) {
	// �ùٸ� �Է����� üũ
	if (x.size() != input_dim) {
		cout << "Perceptron error: x.size() != input_dim.\n";
		exit(-1);
	}

	// forward ���� ����
	double result = 0;

	// �� input�� weight�� ���� ���� ���� ���Ѵ�.
	for (int i = 0; i < input_dim; i++)
		result += x[i] * weights[i];
	result += weights[input_dim];	// Threashold ��

	// net ���� �����صд�. (���� delta �� ���ϴ� �� Ȱ��)
	net = result;

	// activation function
	result = activate(result);

	return result;
}

// weight update
// x: input���� ������ vector, delta_bar: ���� layer���� �Ѿ�� delta_bar
// ��ȯ��: ���� layer�� �ѱ� delta_bar�� ����ϱ� ���� ���� ��� vector (w * delta)
vector<double> Perceptron::update_weight(const vector<double> &x, const double &delta_bar) {
	// delta ���
	delta = delta_bar * SigmoidPrime(net);

	// ���� layer�� �Ѱ��� ���� ��� (w * delta)
	vector<double> rtn;
	for (int i = 0; i < input_dim; i++)
		rtn.push_back(weights[i] * delta);

	// weight update
	for (int i = 0; i < input_dim; i++)
		weights[i] += (-learning_rate * delta * x[i]);

	// threshold update
	weights[input_dim] += (-learning_rate * delta);

	return rtn;
}

// weight�� ������ ������ �ʱ�ȭ�Ѵ�.
void Perceptron::weight_random_initialize() {
	// random number generator ����
	random_device rd;
	mt19937 e2(rd());
	uniform_real_distribution<> dist(-1, 1);

	// random number�� weight �ʱ�ȭ
	weights.resize(input_dim + 1);		// �ܺο� ������ �ʴ� weight �ϳ� �߰� (threshold)
	for (int i = 0; i < input_dim + 1; i++)
		weights[i] = dist(e2);
}

// Ȱ��ȭ �Լ�
double Perceptron::activate(double in) {
	return Sigmoid(in);		// Sigmoid �Լ� ����
//	return hard_limiting(in);
}
