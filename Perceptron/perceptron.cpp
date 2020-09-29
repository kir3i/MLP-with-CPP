// ��ǻ�Ͱ��к� 2016920029 ���ÿ�
// �ΰ����� ���� #2 �Է��� n������ perceptron learning ����
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <string>

using namespace std;

// Perceptron class
class Perceptron {
public:
	// ������
	Perceptron(int n, double lr) {
		this->input_dim = n;				// �Է� ���� �ʱ�ȭ
		this->weight_random_initialize();	// weight �ʱ�ȭ
		this->learning_rate = lr;			// learning_rate �ʱ�ȭ
	}		

	// weight�� �ʱ�ȭ�Ѵ�.
	void weight_random_initialize() {
		// random number generator ����
		random_device rd;
		mt19937 e2(rd());
		uniform_real_distribution<> dist(-1, 1);

		// random number�� weight �ʱ�ȭ
		weights.resize(input_dim + 1);		// �ܺο� ������ �ʴ� weight �ϳ� �߰� (threshold)
		for (int i = 0; i < input_dim + 1; i++)
			weights[i] = dist(e2);
	}

	// weight�� ���ϴ� ������ �����Ѵ�.
	// w: �����ϰ��� �ϴ� weight ���� ���� vector
	void set_weights(const vector<double> &w) {

		// �ùٸ� �Է����� üũ
		if (w.size() != input_dim) {
			cout << "Perceptron�� input_dim�� w.size()�� ��ġ���� �ʽ��ϴ�.\n";
			return;
		}

		// �Է°��� weights�� ����
		for (int i = 0; i < input_dim; i++)
			weights[i] = w[i];
	}

	// weight�� ����Ѵ�.
	void print_weights() {
		cout << "weights: ";
		for (const double &w : weights) {
			cout << w << " ";
		}
		cout << "\n";
	}

	// weight�� ���� ������ �������� ����Ѵ�.
	// (2���� �Է¿��� ��ȿ��)
	void print_linear_function() {
		// �Է� ���� üũ
		if (input_dim != 2)
			return;

		// ������ ������ ���
		cout << "������ ������: x2 = ";
		cout << (weights[0] / weights[1] > 0 ? "-" : "") << abs(weights[0] / weights[1]) << " * x1";
		cout << (weights[2] / weights[1] > 0 ? " - " : " + ") << abs(weights[2] / weights[1]) << "\n";
	}

	// ������ �������� ���Ϸ� �����Ѵ�.
	void write_linear_function(ofstream &outFile) {
		// �Է� ���� üũ
		if (input_dim != 2)
			return;

		// ������ ������ ����
		outFile << (-1 * weights[0] / weights[1]) << " " << (-1 * weights[2] / weights[1]) << "\n";
	}

	// Ȱ��ȭ �Լ�
	double activate(double in) {
		// Hard Limiting
		if (in <= THRESHOLD)
			return 0;
		else
			return 1;
	}

	// forward ����
	// x: input���� ������ vector
	// ��ȯ��: ���� ���
	double foward(const vector<double> &x) {
		// �ùٸ� �Է����� üũ
		if (x.size() != input_dim) {
			cout << "Perceptron�� input_dim�� input_vals.size()�� ��ġ���� �ʽ��ϴ�.\n";
			exit(-1);
		}

		// forward ���� ����
		double result = 0;
		// �� input�� weight�� ���� ���� ���� ���Ѵ�.
		for (int i = 0; i < input_dim; i++)
			result += x[i] * weights[i];
		result += weights[input_dim];	// Threashold ��
		// activation function
		result = activate(result);
		
		return result;
	}

	// weight update(learning), �� ���� case�� ���� ������Ʈ�Ѵ�.
	// back propagation�� ���� weight�� �����Ѵ�.
	// x: input���� ������ vector, y: �����, target: �ùٸ� �����
	void update_weight(const vector<double> &x, const double &y, const double &target) {
		for (int i = 0; i < input_dim; i++)
			weights[i] += learning_rate * (target - y)*x[i];
		weights[input_dim] += learning_rate * (target - y);
	}

	// weight update(learning), ���� ���� case�� ���� ������Ʈ�Ѵ�.
	// back propagation�� ���� weight�� �����Ѵ�.
	// x: input���� ������ vector, y: �����, target: �ùٸ� �����
	void update_weight(const vector<vector<double>> &input, const vector<double> &y, const vector<double> &target) {
		for (int i = 0; i < input.size(); i++)
			update_weight(input[i], y[i], target[i]);
	}

	// Perceptron �۵� (����)
	// input: input���� ������ vector, target: �ùٸ� �����
	// ��ȯ��: Perceptron ���� ����� ���� loss ��
	double run(const vector<vector<double>> &input, const vector<double> &target) {
		// �ùٸ� �Է����� üũ
		if (input.size() != target.size()) {
			cout << "input�� target�� ������ ��ġ���� �ʽ��ϴ�.\n";
			exit(-1);
		}

		// Perceptron ���� ����
		double loss = 0;
		vector<double> y(input.size());
		// case���� ����
		for (int i = 0; i < input.size(); i++) {
			y[i] = foward(input[i]);								// forward ����
			loss += (target[i] - y[i]) * (target[i] - y[i]) / 2;	// mean squared error
			//update_weight(input[i], y[i], target[i]);
		}
		update_weight(input, y, target);							// backward ����
		return loss;
	}

private:
	int input_dim = 0;				// input ����
	double learning_rate;			// learning_rate
	vector<double> weights;			// perceptron�� weight ��
	const static int THRESHOLD = 0;
};

// main �Լ�
int main(void) {
	const int AND = 1;
	const int OR = 2;
	const int XOR = 3;

	// �Է� ������ 2�� �����Ѵ�.
	int N = 2;	//cin >> N;
	if (N <= 0)	return 0;

	// learning rate �Է�
	double lr;	
	cout << "learning rate�� �Է��ϼ���: ";		cin >> lr;

	// �׽�Ʈ�� gate ����
	int select = 0;
	cout << "�׽�Ʈ�� gate�� ������\n";
	cout << "1: AND, 2: OR, 3: XOR\n";
	cout << "�Է�: ";	cin >> select;
	string filename;

	// Perceptron ����
//	Perceptron p = Perceptron(N, lr);
	

	//�׽�Ʈ�� gate�� ���� input, target ����
	vector<vector<double>> x = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
	vector<double> target;
	switch (select) {
	case AND:
		target = { {0}, {0}, {0}, {1} };
		filename = "AND";
		break;
	case OR:
		target = { {0}, {1}, {1}, {1} };
		filename = "OR";
		break;
	case XOR:
		target = { {0}, {1}, {1}, {0} };
		filename = "XOR";
		break;
	default:
		cout << "Gate�� �߸� �����߽��ϴ�.\n";
		exit(-1);
	}

	// �׽�Ʈ ����
	cout << "\n====================������====================\n";
	
	ofstream epochFile(filename + "_lr=" + to_string(lr) + "_epoch.txt");
//	ofstream lineFile(filename + ".txt");
//	ofstream lossFile(filename + "_loss.txt");
	for (int i = 0; i < 1000000; i++) {
		Perceptron p = Perceptron(N, lr);
		double loss = 1;
		int epoch;
		if (i%100000 == 0)
			cout << i << "\n";
		for (epoch = 1; loss != 0; epoch++) {
			//cout << "epoch: " << epoch << "\n";
			//p.print_weights();					// weight ���
			//p.print_linear_function();			// ������ ������ ���
			//p.write_linear_function(lineFile);	// ������ ������ ����
			loss = p.run(x, target);			// perceptron ����
	//		cout << "loss: " << loss << "\n\n";	// loss ���
	//		lossFile << loss << "\n";			// loss ����
			
			// epoch�� 10��ȸ�� ������ ����
			if (epoch == 100000)
				break;
		}
		epochFile << epoch << "\n";
	}
	epochFile.close();
//	lineFile.close();
//	lossFile.close();
	return 0;
}