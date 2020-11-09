#pragma once

using namespace std;

// Perceptron class
class Perceptron {
public:
	// Perceptron ������
	// n: �Է� ����, lr: learning rate
	Perceptron(int n, double lr);

	// weight�� ��ȯ�Ѵ�.
	vector<double> get_weights();

	// weight�� �ֿܼ� ����Ѵ�.
	void print_weights();

	// weight�� ���� ������ �������� ����Ѵ�.
	// 2���� �Է¿��� ��ȿ
	void print_linear_function();

	// ������ �������� ���Ϸ� �����Ѵ�.
	// 2���� �Է¿��� ��ȿ
	// outFile: ������ �������� ����� ����
	void write_linear_function(ofstream &outFile);

	// forward ����
	// x: input���� ������ vector
	// ��ȯ��: forward ���� ���
	double forward(const vector<double> &x);

	// weight update
	// x: input���� ������ vector, delta_bar: ���� layer���� �Ѿ�� delta_bar
	// ��ȯ��: ���� layer�� �ѱ� delta_bar�� ����ϱ� ���� ���� ��� vector (w * delta)
	vector<double> update_weight(const vector<double> &x, const double &delta_bar);

private:
	int input_dim = 0;				// input ����
	double learning_rate;			// learning_rate
	vector<double> weights;			// perceptron�� weight ��
	double net;						// net�� (activation �Լ� ���� ��)
	double delta;					// delta��, ���� layer���� �Ѿ�� delta_bar�� ���

	// weight�� ������ ������ �ʱ�ȭ�Ѵ�.
	void weight_random_initialize();

	// Ȱ��ȭ �Լ�
	double activate(double in);
};