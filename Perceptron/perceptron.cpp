// 컴퓨터과학부 2016920029 유시온
// 인공지능 과제 #1 Perceptron 구현
#include <iostream>
#include <random>
#include <vector>

using namespace std;

// Perceptron class
class Perceptron {
public:
	// 생성자
	Perceptron(int n) {
		this->input_dim = n;		// 입력 차원 초기화
		this->weight_random_initialize();	// weight 초기화
	}

	// weight를 초기화한다.
	void weight_random_initialize() {
		// random number generator 정의
		random_device rd;
		mt19937 e2(rd());
		uniform_real_distribution<> dist(-10, 10);

		// random number로 weight 초기화
		weights.clear();
		for (int i = 0; i < input_dim; i++)
			weights.push_back(dist(e2));
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

	// input 차원이 얼마인지 리턴한다.
	void get_input_dim() {
		cout << this->input_dim;
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
	// 반환값: Perceptron 실행 결과
	double run(const vector<double> &input_vals) {
		// 올바른 입력인지 체크
		if (input_vals.size() != input_dim) {
			cout << "Perceptron의 input_dim과 input_vals.size()가 일치하지 않습니다.\n";
			return ERROR;
		}

		// Perceptron 연산 시작
		double result = 0;

		// 각 input에 weight를 곱한 값의 합을 구한다.
		for (int i = 0; i < input_dim; i++)
			result += input_vals[i] * weights[i];

		// activation function
		result = activate(result);

		return result;
	}

	// AND gate를 테스트한다.
	// 반환값: 틀린 케이스 개수를 리턴한다.
	int test_ANDgate() {
		int ret = 0;	// 틀린 케이스(WA) 개수

		// AND gate를 테스트하기 위한 input과 answer 정의 (x0는 1로 고정)
		vector<double> test_input[4] = { {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1} };
		int ans[4] = { 0, 0, 0, 1 };

		// test 수행
		for (int i = 0; i < 4; i++) {
			if (ans[i] != run(test_input[i]))
				ret++;	// 실행 결과가 틀린 경우 1 증가
		}

		return ret;
	}

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

private:
	int input_dim = 0;				// input 차원
	vector<double> weights;			// perceptron의 weight 값
	const static int ERROR = -1;
	const static int THRESHOLD = 0;
};

// main 함수
int main(void) {

	// AND gate를 테스트 할 목적이므로 N = 3으로 고정
	int N = 3;	//cin >> N;
	if (N <= 0)	return 0;

	// Perceptron 생성
	Perceptron p = Perceptron(N);

	// chk_WA: 한 case에서 WA 개수
	// cnt: 테스트 반복 횟수
	int chk_WA = 4, cnt = 0;

	// 테스트
	while (true) {
		cout << "#" << (++cnt) << "\n";
		chk_WA = p.test_ANDgate_print();	// test
		cout << "WA개수: " << chk_WA << "\n\n";

		// WA가 없는 경우 (다 맞은 경우)
		if (chk_WA == 0)
			break;

		// weight 입력
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