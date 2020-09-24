// 컴퓨터과학부 2016920029 유시온
// 인공지능 과제 #2 입력이 n차원인 perceptron learning 구현
#include <iostream>
#include <random>
#include <vector>

using namespace std;

// Perceptron class
class Perceptron {
public:
	// 생성자
	Perceptron(int n, double lr) {
		this->input_dim = n;		// 입력 차원 초기화
		this->weight_random_initialize();	// weight 초기화
		this->learning_rate = lr;
	}

	// weight를 초기화한다.
	void weight_random_initialize() {
		// random number generator 정의
		random_device rd;
		mt19937 e2(rd());
		uniform_real_distribution<> dist(-1, 1);

		// random number로 weight 초기화
		weights.resize(input_dim + 1);	// 외부엔 보이지 않는 weight 하나 추가 (threshold)
		for (int i = 0; i < input_dim + 1; i++)
			weights[i] = dist(e2);
	}

	// weight를 원하는 값으로 설정한다.
	// w: 설정하고자 하는 weight 값을 갖는 vector
	void set_weights(const vector<double> &w) {

		// 올바른 입력인지 체크
		if (w.size() != input_dim) {
			cout << "Perceptron의 input_dim과 w.size()가 일치하지 않습니다.\n";
			return;
		}

		// 입력값을 weights로 복사
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
		cout << "직선의 방정식: x2 = ";
		cout << (weights[0] / weights[1] > 0 ? "-" : "") << abs(weights[0] / weights[1]) << " * x1";
		cout << (weights[2] / weights[1] > 0 ? " - " : " + ") << abs(weights[2] / weights[1]) << "\n";
	}

	// 활성화 함수
	double activate(double in) {
		// Hard Limiting
		if (in <= THRESHOLD)
			return 0;
		else
			return 1;
	}

	// Perceptron 연산(작동)
	// input_vals: input값을 저장한 vector
	// 반환값: 연산 결과
	double foward(const vector<double> &x) {
		// 올바른 입력인지 체크
		if (x.size() != input_dim) {
			cout << "Perceptron의 input_dim과 input_vals.size()가 일치하지 않습니다.\n";
			exit(-1);
		}

		// Perceptron 연산 시작
		double result = 0;

		// 각 input에 weight를 곱한 값의 합을 구한다.
		for (int i = 0; i < input_dim; i++)
			result += x[i] * weights[i];
		result += weights[input_dim];	// Threashold 값
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

	//error 리턴
	double run(const vector<vector<double>> &input, const vector<double> &target) {
		if (input.size() != target.size()) {
			cout << "input과 target의 개수가 일치하지 않습니다.\n";
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
	int input_dim = 0;				// input 차원
	double learning_rate;
	vector<double> weights;			// perceptron의 weight 값
	const static int THRESHOLD = 0;
};

// main 함수
int main(void) {
	const int AND = 1;
	const int OR = 2;
	const int XOR = 3;

	int N = 2;	//cin >> N;
	if (N <= 0)	return 0;
	double lr;	
	cout << "learning rate를 입력하세요: ";		cin >> lr;
	int select = 0;
	cout << "테스트할 gate를 고르세요\n";
	cout << "1: AND, 2: OR, 3: XOR\n";
	cout << "입력: ";	cin >> select;

	// Perceptron 생성
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
		cout << "Gate를 잘못 선택했습니다.\n";
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
		
		cout << "정답률: " << ok << "/" << x.size() << "\n";
	}
	*/
	// test
	cout << "\n====================실행결과====================\n";
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