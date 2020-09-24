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
		uniform_real_distribution<> dist(-10, 10);

		// random number로 weight 초기화
		weights.resize(input_dim + 1);	// 외부엔 보이지 않는 weight 하나 추가
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
	// 반환값: 올바른 예측을 했다면 1, 아니라면 0
	int run(const vector<double> &x, const double &target) {
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
		
		return result == target? 1: 0;
	}
/*
	// 테스트 결과를 출력하면서 테스트를 수행한다.
	// 반환값: 틀린 케이스 개수를 리턴한다.
	int test_ANDgate_print() {
		int ret = 0;	// 틀린 케이스(WA) 개수

		// AND gate를 테스트하기 위한 input과 answer 정의
		vector<double> test_input[4] = { {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1} };
		int ans[4] = { 0, 0, 0, 1 };

		// test 수행 및 결과 출력
		for (int i = 0; i < 4; i++) {
			double result = run(test_input[i]);	// 실행 결과
			// 실행 결과 출력
			cout << test_input[i][1] << " " << test_input[i][2] << " --> ";
			cout << result;

			// 실행 결과가 틀린 경우
			if (ans[i] != result) {
				ret++;				// 1 증가
				cout << " == WA";	// 틀린 케이스(WA)임을 표시
			}
			cout << "\n";
		}
		// 직선의 방정식 출력
		cout << "직선의 방정식: x2 = ";
		cout << (weights[1] / weights[2] > 0 ? "-" : "") << abs(weights[1] / weights[2]) << " * x1";
		cout << (weights[0] / weights[2] > 0 ? " - " : " + ") << abs(weights[0] / weights[2]) << "\n";
		return ret;
	}
*/
private:
	int input_dim = 0;				// input 차원
	double learning_rate;
	vector<double> weights;			// perceptron의 weight 값
	const static int THRESHOLD = 0;
};

// main 함수
int main(void) {

	int N = 2;	//cin >> N;
	if (N <= 0)	return 0;

	// Perceptron 생성
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
		
		cout << "정답률: " << ok << "/" << x.size() << "\n";
	}

	return 0;
}