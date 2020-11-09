#include <iostream>
#include <fstream>
#include <vector>
#include "Perceptron.h"
#include "Layer.h"

using namespace std;

// ������
// input_dim: layer�� �Է� ����, output_dim: layer�� ��� ����
// lr: learning rate
Layer::Layer(int input_dim, int output_dim, double lr) {
	this->input_dim = input_dim;			// �Է� ���� �ʱ�ȭ
	this->output_dim = output_dim;			// ��� ���� �ʱ�ȭ

	// layer�� ������ perceptron ����
	for (int i = 0; i < output_dim; i++)
		nodes.push_back(Perceptron(input_dim, lr));
}

// Layer�� ���ϴ� node���� weight�� file�� ���
// outFile: weight�� ������ file
// ��������
// - Layer ���� �ϳ��� ����� �����Ѵ�.
// - �� row�� �� ����� weight�� ��Ÿ����. bias�� ���� �������� ��ϵȴ�.
void Layer::write_weight(ofstream &outFile) {
	for (Perceptron &n : nodes) {
		for (const double &w : n.get_weights())
			outFile << w << " ";
		outFile << "\n";
	}
}

// layer�� ���� nodes���� ������ �������� �ֿܼ� ����Ѵ�.
// ǥ������: [���̾� ��ȣ]-[��� ��ȣ] ������ ������: [������ ������]
// layer_num: layer�� ��ȣ (�� ��° layer����)
void Layer::print_linear_function(const int &layer_num) {
	for (int i = 0; i < nodes.size(); i++) {
		cout << layer_num << "-" << i + 1 << " ";
		nodes[i].print_linear_function();
	}
}

// layer�� ���� nodes���� ������ �������� file�� �����Ѵ�.
// outFile: ������ �������� ������ ����
void Layer::write_linear_function(ofstream &outFile) {
	for (Perceptron &n : nodes)
		n.write_linear_function(outFile);
	outFile << "\n";
}

// ���� forward ���꿡�� ����ߴ� input�� ��ȯ�Ѵ�.
vector<double> Layer::get_prev_x() {
	return prev_x;
}

// foward ����
// x: layer�� ���� �Է�
// ��ȯ��: forward ���� ���
vector<double> Layer::forward(const vector<double> &x) {
	// �ùٸ� �Է����� üũ
	if (x.size() != input_dim) {
		cout << "Layer error: x.size() != input_dim\n";
		cout << "input.size() == " << x.size() << "\n";
		cout << "input_dim == " << input_dim << "\n";
		exit(-1);
	}

	prev_x = x;			// �Է��� �����صд�. (backward���� ���)

	// forward ���� ����
	vector<double> rtn;
	// layer�� ���� perceptron ���� forward ����
	for (Perceptron &p : nodes)
		rtn.push_back(p.forward(x));

	return rtn;
}

// backward ����
// delta_bar: ���� layer ���� ��� �� delta_bar vector
// ��ȯ��: ���� layer�� ���� delta_bar vector
vector<double> Layer::backward(const vector<double> &delta_bar) {
	// �ùٸ� �Է����� üũ
	if (delta_bar.size() != output_dim) {
		cout << "Layer error: delta_bar.size() != output_dim\n";
		exit(-1);
	}

	// backward ���� ����
	vector<double> rtn(input_dim, 0);		// ���� layer�� ���� delta_bar vector

	// layer�� ���� perceptron�� ���� delta_bar�� �̿��� weight update
	for (int i = 0; i < output_dim; i++) {
		vector<double> update_delta = nodes[i].update_weight(prev_x, delta_bar[i]);

		// ���� layer�� ���� delta_bar ���
		for (int j = 0; j < input_dim; j++)
			rtn[j] += update_delta[j];
	}

	return rtn;
}
