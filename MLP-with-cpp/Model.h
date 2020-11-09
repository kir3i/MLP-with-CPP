#pragma once

using namespace std;

// Model class
class Model {
public:
	// ������
	// input_dim: �Է� data�� ����, layers_dim: �� layer�� output_dim�� ���� ����
	// lr: learning rate
	Model(int input_dim, const vector<int> &layers_dim, double lr);

	// Model�� �� Layer�� node���� weight�� file�� ����Ѵ�.
	// outFile: weight�� ����� file
	// ��������
	// - Layer ���� �ϳ��� ����� �����Ѵ�.
	// - �� row�� �� ����� weight�� ��Ÿ����. bias�� ���� ������ ����̴�.
	void write_weights(ofstream &outFile);

	// model�� ���� layer�鿡 ���� �� node�� ������ �������� �ֿܼ� ����Ѵ�.
	// �������: [layer ��ȣ]-[node��ȣ] ������ ������: [������ ������]
	// layer��ȣ�� node ��ȣ ��� 1������ �����Ѵ�.
	void print_linear_function();

	// model�� ���� layer�鿡 ���� �� node�� ������ �������� file�� �����Ѵ�.
	// �� epoch�� ","�� �����Ѵ�.
	// outfile: ������ �������� ������ file
	void write_linear_function(ofstream &outFile);

	// �� �Է��� hidden layer�� ���� �̵��� file�� ����Ѵ�.
	// {2, 1}�� �ش��ϴ� �𵨿��� ���������� �۵��Ѵ�.
	// x: model�� ���� �Է�, outFile: ����� ����
	// �� epoch�� "\n\n"���� �����Ѵ�.
	// ���������� (0, 0), (0, 1), (1, 0), (1, 1)�� �̵��� ���� ����Ѵ�.
	void write_dot_moving(const vector<double> &x, ofstream &outFile);

	// forward ����
	// x: model�� ���� �Է�
	// ��ȯ��: forward ���� ���
	vector<double> forward(const vector<double> &x);

	// backward ����
	// x: �Է� data, y: model�� ������ ���, target: ���� ���
	void backward(const vector<double> &x, const vector<double> &y, const vector<double> &target);

	// model �۵�, �� case�� ���ؼ� �н�
	// x: model�� ���� input, target: ���� ���
	// ��ȯ��: model�� ���� ��� (���� ���)
	vector<double> run(const vector<double> &x, const vector<double> &target);

	// model �۵�, �ټ��� case�� ���ؼ� �н�
	// x: model�� ���� input, target: ���� ���
	// outFile: dot_moving�� ����� ����
	// print: �ش� epoch���� ����� ���, dot_moving: �� �̵� ��� ��� ({2, 1} �𵨿����� �۵�)
	// ��ȯ��: ������ ����� loss��
	double run(const vector<vector<double>> &input, const vector<vector<double>> &target, ofstream &outFile, const bool &print = true, const bool &dot_moving = true);

private:
	vector<Layer> layers;	// model�� �����ϴ� layer vector
};