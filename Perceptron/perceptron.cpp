// ��ǻ�Ͱ��к� 2016920029 ���ÿ�
// �ΰ����� ���� #2 �Է��� n������ perceptron learning ����
#include <iostream>
#include <random>
#include <vector>

using namespace std;

// Perceptron class
class Perceptron {
public:
	// ������
	Perceptron(int n, double lr) {
		this->input_dim = n;		// �Է� ���� �ʱ�ȭ
		this->weight_random_initialize();	// weight �ʱ�ȭ
		this->learning_rate = lr;
	}

	// weight�� �ʱ�ȭ�Ѵ�.
	void weight_random_initialize() {
		// random number generator ����
		random_device rd;
		mt19937 e2(rd());
		uniform_real_distribution<> dist(-1, 1);

		// random number�� weight �ʱ�ȭ
		weights.resize(input_dim + 1);	// �ܺο� ������ �ʴ� weight �ϳ� �߰� (threshold)
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

	void print_weights() {
		cout << "weights: ";
		for (const double &w : weights) {
			cout << w << " ";
		}
		cout << "\n";
	}

	void print_linear_function() {
		cout << "������ ������: x2 = ";
		cout << (weights[0] / weights[1] > 0 ? "-" : "") << abs(weights[0] / weights[1]) << " * x1";
		cout << (weights[2] / weights[1] > 0 ? " - " : " + ") << abs(weights[2] / weights[1]) << "\n";
	}

	// Ȱ��ȭ �Լ�
	double activate(double in) {
		// Hard Limiting
		if (in <= THRESHOLD)
			return 0;
		else
			return 1;
	}

	// Perceptron ����(�۵�)
	// input_vals: input���� ������ vector
	// ��ȯ��: ���� ���
	double foward(const vector<double> &x) {
		// �ùٸ� �Է����� üũ
		if (x.size() != input_dim) {
			cout << "Perceptron�� input_dim�� input_vals.size()�� ��ġ���� �ʽ��ϴ�.\n";
			exit(-1);
		}

		// Perceptron ���� ����
		double result = 0;

		// �� input�� weight�� ���� ���� ���� ���Ѵ�.
		for (int i = 0; i < input_dim; i++)
			result += x[i] * weights[i];
		result += weights[input_dim];	// Threashold ��
		// activation function
		result = activate(result);
		
		return result;
	}

	void update_weight(const vector<double> &x, const double &y, const double &target) {
		for (int i = 0; i < input_dim; i++)
			weights[i] += learning_rate * (target - y)*x[i];
		weights[input_dim] += learning_rate * (target - y);
	}

	void update_weight(const vector<vector<double>> &input, const vector<double> &y, const vector<double> &target) {
		for (int i = 0; i < input.size(); i++)
			update_weight(input[i], y[i], target[i]);
	}

	//error ����
	double run(const vector<vector<double>> &input, const vector<double> &target) {
		if (input.size() != target.size()) {
			cout << "input�� target�� ������ ��ġ���� �ʽ��ϴ�.\n";
			exit(-1);
		}
		double loss = 0;
		vector<double> y(input.size());
		for (int i = 0; i < input.size(); i++) {
			y[i] = foward(input[i]);
			loss += (target[i] - y[i]) * (target[i] - y[i]) / 2;
			//update_weight(input[i], y[i], target[i]);
		}
		update_weight(input, y, target);
		return loss;
	}

private:
	int input_dim = 0;				// input ����
	double learning_rate;
	vector<double> weights;			// perceptron�� weight ��
	const static int THRESHOLD = 0;
};

// main �Լ�
int main(void) {
	const int AND = 1;
	const int OR = 2;
	const int XOR = 3;

	int N = 2;	//cin >> N;
	if (N <= 0)	return 0;
	double lr;	
	cout << "learning rate�� �Է��ϼ���: ";		cin >> lr;
	int select = 0;
	cout << "�׽�Ʈ�� gate�� ������\n";
	cout << "1: AND, 2: OR, 3: XOR\n";
	cout << "�Է�: ";	cin >> select;

	// Perceptron ����
	Perceptron p = Perceptron(N, lr);

	//input 
	vector<vector<double>> x = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
	vector<double> target;
	switch (select) {
	case AND:
		target = { {0}, {0}, {0}, {1} };
		break;
	case OR:
		target = { {0}, {1}, {1}, {1} };
		break;
	case XOR:
		target = { {0}, {1}, {1}, {0} };
		break;
	default:
		cout << "Gate�� �߸� �����߽��ϴ�.\n";
		exit(-1);
	}

	//test
	/*
	for (int ok = 0, epoch = 0; ok < x.size(); epoch++) {
		ok = 0;
		cout << "epoch: " << epoch << "\n";
		for (int i = 0; i < x.size(); i++)
			ok += ((target[i] == p.forward(x[i]))? 1: 0);
		p.weight_random_initialize();
		
		cout << "�����: " << ok << "/" << x.size() << "\n";
	}
	*/
	// test
	cout << "\n====================������====================\n";
	double loss = 1;
	for (int epoch = 1; loss != 0 ; epoch++) {
		cout << "epoch: " << epoch << "\n";
		p.print_weights();
		p.print_linear_function();
		loss = p.run(x, target);
		cout << "loss: " << loss << "\n\n";
	}
	return 0;
}