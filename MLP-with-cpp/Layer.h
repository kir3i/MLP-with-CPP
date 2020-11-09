#pragma once
#include "Perceptron.h"

using namespace std;

// Layer Class
class Layer {
public:
	// ������
	// input_dim: layer�� �Է� ����, output_dim: layer�� ��� ����
	// lr: learning rate
	Layer(int input_dim, int output_dim, double lr);

	// Layer�� ���ϴ� node���� weight�� file�� ���
	// outFile: weight�� ������ file
	// ��������
	// - Layer ���� �ϳ��� ����� �����Ѵ�.
	// - �� row�� �� ����� weight�� ��Ÿ����. bias�� ���� �������� ��ϵȴ�.
	void write_weight(ofstream &outFile);

	// layer�� ���� nodes���� ������ �������� �ֿܼ� ����Ѵ�.
	// ǥ������: [���̾� ��ȣ]-[��� ��ȣ] ������ ������: [������ ������]
	// layer_num: layer�� ��ȣ (�� ��° layer����)
	void print_linear_function(const int &layer_num);

	// layer�� ���� nodes���� ������ �������� file�� �����Ѵ�.
	// outFile: ������ �������� ������ ����
	void write_linear_function(ofstream &outFile);

	// ���� forward ���꿡�� ����ߴ� input�� ��ȯ�Ѵ�.
	vector<double> get_prev_x();

	// foward ����
	// x: layer�� ���� �Է�
	// ��ȯ��: forward ���� ���
	vector<double> forward(const vector<double> &x);

	// backward ����
	// delta_bar: ���� layer ���� ��� �� delta_bar vector
	// ��ȯ��: ���� layer�� ���� delta_bar vector
	vector<double> backward(const vector<double> &delta_bar);

private:
	vector<Perceptron> nodes;	// layer�� �����ϴ� perceptron vector
	int input_dim;				// �Է� ����
	int output_dim;				// ��� ����
	vector<double> prev_x;		// forward ���꿡 ����� �Է�
};