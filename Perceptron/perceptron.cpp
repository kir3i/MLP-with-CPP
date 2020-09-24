// ��ǻ�Ͱ��к� 2016920029 ���ÿ�
// �ΰ����� ���� #1 Perceptron ����
#include <iostream>
#include <random>
#include <vector>

using namespace std;

// Perceptron class
class Perceptron {
public:
	// ������
	Perceptron(int n) {
		this->input_dim = n;		// �Է� ���� �ʱ�ȭ
		this->weight_random_initialize();	// weight �ʱ�ȭ
	}

	// weight�� �ʱ�ȭ�Ѵ�.
	void weight_random_initialize() {
		// random number generator ����
		random_device rd;
		mt19937 e2(rd());
		uniform_real_distribution<> dist(-10, 10);

		// random number�� weight �ʱ�ȭ
		weights.clear();
		for (int i = 0; i < input_dim; i++)
			weights.push_back(dist(e2));
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

	// input ������ ������ �����Ѵ�.
	void get_input_dim() {
		cout << this->input_dim;
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
	// ��ȯ��: Perceptron ���� ���
	double run(const vector<double> &input_vals) {
		// �ùٸ� �Է����� üũ
		if (input_vals.size() != input_dim) {
			cout << "Perceptron�� input_dim�� input_vals.size()�� ��ġ���� �ʽ��ϴ�.\n";
			return ERROR;
		}

		// Perceptron ���� ����
		double result = 0;

		// �� input�� weight�� ���� ���� ���� ���Ѵ�.
		for (int i = 0; i < input_dim; i++)
			result += input_vals[i] * weights[i];

		// activation function
		result = activate(result);

		return result;
	}

	// AND gate�� �׽�Ʈ�Ѵ�.
	// ��ȯ��: Ʋ�� ���̽� ������ �����Ѵ�.
	int test_ANDgate() {
		int ret = 0;	// Ʋ�� ���̽�(WA) ����

		// AND gate�� �׽�Ʈ�ϱ� ���� input�� answer ���� (x0�� 1�� ����)
		vector<double> test_input[4] = { {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1} };
		int ans[4] = { 0, 0, 0, 1 };

		// test ����
		for (int i = 0; i < 4; i++) {
			if (ans[i] != run(test_input[i]))
				ret++;	// ���� ����� Ʋ�� ��� 1 ����
		}

		return ret;
	}

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

private:
	int input_dim = 0;				// input ����
	vector<double> weights;			// perceptron�� weight ��
	const static int ERROR = -1;
	const static int THRESHOLD = 0;
};

// main �Լ�
int main(void) {

	// AND gate�� �׽�Ʈ �� �����̹Ƿ� N = 3���� ����
	int N = 3;	//cin >> N;
	if (N <= 0)	return 0;

	// Perceptron ����
	Perceptron p = Perceptron(N);

	// chk_WA: �� case���� WA ����
	// cnt: �׽�Ʈ �ݺ� Ƚ��
	int chk_WA = 4, cnt = 0;

	// �׽�Ʈ
	while (true) {
		cout << "#" << (++cnt) << "\n";
		chk_WA = p.test_ANDgate_print();	// test
		cout << "WA����: " << chk_WA << "\n\n";

		// WA�� ���� ��� (�� ���� ���)
		if (chk_WA == 0)
			break;

		// weight �Է�
		vector<double> w;
		cout << "Input new weights (" << N << " values): ";
		for (int i = 0; i < N; i++) {
			double x;	cin >> x;
			w.push_back(x);
		}
		p.set_weights(w);	// weight setting
	}

	return 0;
}