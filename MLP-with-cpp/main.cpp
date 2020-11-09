// ��ǻ�Ͱ��к� 2016920029 ���ÿ�
// �ΰ����� ���� #3 Multi-Layer perceptron ����
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "activation_functions.h"
#include "objective_functions.h"
#include "Perceptron.h"
#include "Layer.h"
#include "Model.h"

using namespace std;

// main �Լ�
int main(void) {
	const double TOLERANCE = 0.001;		// tolerance
	const int AND = 1;					// AND 
	const int OR = 2;					// OR
	const int XOR = 3;					// XOR
	const int DONUT = 4;				// DONUT

	// �׽�Ʈ�� data ����
	int select = 0;
	cout << "�׽�Ʈ�� data�� ������\n";
	cout << "1: AND, 2: OR, 3: XOR 4: DONUT\n";
	cout << "�Է�: ";	cin >> select;
	string filename;

	//�׽�Ʈ�� data�� ���� input, target ����
	vector<vector<double>> x;
	vector<vector<double>> target;
	switch (select) {
	case AND:
		x = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
		target = { {0}, {0}, {0}, {1} };
		filename = "AND";
		break;
	case OR:
		x = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
		target = { {0}, {1}, {1}, {1} };
		filename = "OR";
		break;
	case XOR:
		x = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
		target = { {0}, {1}, {1}, {0} };
		filename = "XOR";
		break;
	case DONUT:
		x = { {0, 0}, {0, 1}, {1, 0}, {1, 1}, {0.5, 1}, {1, 0.5}, {0, 0.5}, {0.5, 0}, {0.5, 0.5} };
		target = { {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {1} };
		filename = "DONUT";
		break;
	default:
		cout << "Data�� �߸� �����߽��ϴ�.\n";
		exit(-1);
	}

	// layer �� ����
	int layer_num = 0;
	cout << "layer ���� ���ϼ���: ";	cin >> layer_num;
	
	// layer ���� ��� ���� ����
	cout << "�� layer�� output ������ ����� �����Ͽ� �Է��ϼ���: ";
	vector<int> layers_dim(layer_num);
	for (int i = 0; i < layer_num; i++)
		cin >> layers_dim[i];

	// �ùٸ� �Է����� üũ
	if (layers_dim[layer_num - 1] != target[0].size()) {
		cout << "���� ��� ������ target�� ��ġ���� �ʽ��ϴ�.\n";
		exit(-1);
	}

	// learning rate �Է�
	double lr;
	cout << "learning rate�� �Է��ϼ���: ";		cin >> lr;

	// model ����
	Model m = Model(x[0].size(), layers_dim, lr);

	// ���� ���� ����� file ����
	ofstream lossFile(filename + "_loss.txt");
	ofstream lineFile(filename + "_line.txt");
	ofstream weightFile(filename + "_weight.txt");
	ofstream dotFile(filename + "_dot.txt");

	// �׽�Ʈ ����
	cout << "\n====================������====================\n";
	double loss = 1;	// loss��

	// model �н� ����
	for (int epoch = 1; loss > TOLERANCE; epoch++) {
		cout << "epoch: " << epoch << "\n";
		loss = m.run(x, target, dotFile, false, false);	// �н� ����

		// ���� ���� ���
		m.print_linear_function();
		m.write_linear_function(lineFile);

		// loss ���
		cout << "loss: " << loss << "\n\n";
		lossFile << loss << "\n";
	}

	// ���� weight ����
	m.write_weights(weightFile);

	lossFile.close();
	lineFile.close();
	weightFile.close();
	dotFile.close();

	return 0;
}