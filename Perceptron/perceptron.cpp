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
		uniform_real_distribution<> dist(-10, 10);

		// random number�� weight �ʱ�ȭ
		weights.resize(input_dim + 1);	// �ܺο� ������ �ʴ� weight �ϳ� �߰�
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
	// ��ȯ��: �ùٸ� ������ �ߴٸ� 1, �ƴ϶�� 0
	int run(const vector<double> &x, const double &target) {
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
		
		return result == target? 1: 0;
	}
/*
	// �׽�Ʈ ����� ����ϸ鼭 �׽�Ʈ�� �����Ѵ�.
	// ��ȯ��: Ʋ�� ���̽� ������ �����Ѵ�.
	int test_ANDgate_print() {
		int ret = 0;	// Ʋ�� ���̽�(WA) ����

		// AND gate�� �׽�Ʈ�ϱ� ���� input�� answer ����
		vector<double> test_input[4] = { {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1} };
		int ans[4] = { 0, 0, 0, 1 };

		// test ���� �� ��� ���
		for (int i = 0; i < 4; i++) {
			double result = run(test_input[i]);	// ���� ���
			// ���� ��� ���
			cout << test_input[i][1] << " " << test_input[i][2] << " --> ";
			cout << result;

			// ���� ����� Ʋ�� ���
			if (ans[i] != result) {
				ret++;				// 1 ����
				cout << " == WA";	// Ʋ�� ���̽�(WA)���� ǥ��
			}
			cout << "\n";
		}
		// ������ ������ ���
		cout << "������ ������: x2 = ";
		cout << (weights[1] / weights[2] > 0 ? "-" : "") << abs(weights[1] / weights[2]) << " * x1";
		cout << (weights[0] / weights[2] > 0 ? " - " : " + ") << abs(weights[0] / weights[2]) << "\n";
		return ret;
	}
*/
private:
	int input_dim = 0;				// input ����
	double learning_rate;
	vector<double> weights;			// perceptron�� weight ��
	const static int THRESHOLD = 0;
};

// main �Լ�
int main(void) {

	int N = 2;	//cin >> N;
	if (N <= 0)	return 0;

	// Perceptron ����
	Perceptron p = Perceptron(N, 0.01);

	//input 
	vector<vector<double>> x = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
	
	//AND gate
	vector<double> target = { {0}, {0}, {0}, {1} };
	//test
	for (int ok = 0, epoch = 0; ok < x.size(); epoch++) {
		ok = 0;
		cout << "epoch: " << epoch << "\n";
		for (int i = 0; i < x.size(); i++)
			ok += p.run(x[i], target[i]);
		p.weight_random_initialize();
		
		cout << "�����: " << ok << "/" << x.size() << "\n";
	}

	return 0;
}