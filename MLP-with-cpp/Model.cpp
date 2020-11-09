#include <vector>
#include <iostream>
#include <fstream>
#include "objective_functions.h"
#include "Layer.h"
#include "Model.h"

using namespace std;

// ������
// input_dim: �Է� data�� ����, layers_dim: �� layer�� output_dim�� ���� ����
// lr: learning rate
Model::Model(int input_dim, const vector<int> &layers_dim, double lr) {
	// �ùٸ� �Է����� üũ
	if (layers_dim.size() == 0) {
		cout << "Model error: layers_dim.size() == 0\n";
		exit(-1);
	}

	// model ����
	layers.push_back(Layer(input_dim, layers_dim[0], lr));
	for (int i = 1; i < layers_dim.size(); i++)
		layers.push_back(Layer(layers_dim[i - 1], layers_dim[i], lr));
}

// Model�� �� Layer�� node���� weight�� file�� ����Ѵ�.
// outFile: weight�� ����� file
// ��������
// - Layer ���� �ϳ��� ����� �����Ѵ�.
// - �� row�� �� ����� weight�� ��Ÿ����. bias�� ���� ������ ����̴�.
void Model::write_weights(ofstream &outFile) {
	for (int i = 0; i < layers.size(); i++) {
		outFile << "Layer " << i + 1 << "\n";
		layers[i].write_weight(outFile);
		outFile << "\n";
	}
}

// model�� ���� layer�鿡 ���� �� node�� ������ �������� �ֿܼ� ����Ѵ�.
// �������: [layer ��ȣ]-[node��ȣ] ������ ������: [������ ������]
// layer��ȣ�� node ��ȣ ��� 1������ �����Ѵ�.
void Model::print_linear_function() {
	for (int i = 0; i < layers.size(); i++)
		layers[i].print_linear_function(i);
}

// model�� ���� layer�鿡 ���� �� node�� ������ �������� file�� �����Ѵ�.
// �� epoch�� ","�� �����Ѵ�.
// outfile: ������ �������� ������ file
void Model::write_linear_function(ofstream &outFile) {
	for (Layer &l : layers)
		l.write_linear_function(outFile);
	outFile << ",";
}

// �� �Է��� hidden layer�� ���� �̵��� file�� ����Ѵ�.
// {2, 1}�� �ش��ϴ� �𵨿��� ���������� �۵��Ѵ�.
// x: model�� ���� �Է�, outFile: ����� ����
// �� epoch�� "\n\n"���� �����Ѵ�.
// ���������� (0, 0), (0, 1), (1, 0), (1, 1)�� �̵��� ���� ����Ѵ�.
void Model::write_dot_moving(const vector<double> &x, ofstream &outFile) {
	// �ùٸ��� ���� ȣ��
	if (layers.size() != 2 || x.size() != 2)
		return;

	// �� ��° ���̾��� �Է��� üũ�Ѵ�.
	vector<double> in = layers[1].get_prev_x();
	if (in.size() != 2) {
		cout << "model error: write_dot_moving�� ������ �Է¿��� �۵��ϴ� �Լ��Դϴ�.\n";
		return;
	}
	outFile << in[0] << " " << in[1] << "\n";
}

// forward ����
// x: model�� ���� �Է�
// ��ȯ��: forward ���� ���
vector<double> Model::forward(const vector<double> &x) {
	// layer ������� �����Ű�� ����
	vector<double> nx = x;
	for (Layer &l : layers)
		nx = l.forward(nx);

	return nx;
}

// backward ����
// x: �Է� data, y: model�� ������ ���, target: ���� ���
void Model::backward(const vector<double> &x, const vector<double> &y, const vector<double> &target) {
	// �ʱ� delta_bar �Ի�
	vector<double> delta_bar;
	for (int i = 0; i < y.size(); i++)
		delta_bar.push_back(-target[i] + y[i]);

	// back propagation ����
	for (int i = (int)layers.size() - 1; i >= 0; i--)
		delta_bar = layers[i].backward(delta_bar);		// delta_bar ������Ʈ
}

// model �۵�, �� case�� ���ؼ� �н�
// x: model�� ���� input, target: ���� ���
// ��ȯ��: model�� ���� ��� (���� ���)
vector<double> Model::run(const vector<double> &x, const vector<double> &target) {
	// forward ���� ����
	vector<double> y = forward(x);

	// �ùٸ� �Է����� üũ
	if (y.size() != target.size()) {
		cout << "Model error: y.size() != target.size()\n";
		exit(-1);
	}

	// backward ���� ����
	backward(x, y, target);

	return y;
}

// model �۵�, �ټ��� case�� ���ؼ� �н�
// x: model�� ���� input, target: ���� ���
// outFile: dot_moving�� ����� ����
// print: �ش� epoch���� ����� ���, dot_moving: �� �̵� ��� ��� ({2, 1} �𵨿����� �۵�)
// ��ȯ��: ������ ����� loss��
double Model::run(const vector<vector<double>> &input, const vector<vector<double>> &target, ofstream &outFile, const bool &print, const bool &dot_moving) {
	// �ùٸ� �Է����� üũ
	if (input.size() != target.size()) {
		cout << "Model error: input.size() != target.size()\n";
		exit(-1);
	}

	// model �۵� ����
	double loss = 0;

	// ���̽� ���� ���
	for (int i = 0; i < input.size(); i++) {
		vector<double> predict = run(input[i], target[i]);	// ������ ���
		loss += mean_squared_error(predict, target[i]);		// loss ���

		// �� �̵� ��� ���
		if (dot_moving)
			write_dot_moving(input[i], outFile);

		// ���� ��� ���
		if (print) {
			for (const double &x : input[i])
				cout << x << " ";
			cout << " -> ";
			for (const double &y : predict)
				cout << y << " ";
			cout << "\n";
		}
	}
	outFile << "\n";

	// loss ����
	return loss;
}