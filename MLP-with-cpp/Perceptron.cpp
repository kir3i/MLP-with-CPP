#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include "activation_functions.h"
#include "Perceptron.h"

using namespace std;


// Perceptron 생성자
// n: 입력 차원, lr: learning rate
Perceptron::Perceptron(int n, double lr) {
	this->input_dim = n;				// 입력 차원 초기화
	this->weight_random_initialize();	// weight 초기화
	this->learning_rate = lr;			// learning_rate 초기화
}

// weight를 반환한다.
vector<double> Perceptron::get_weights() {
	return weights;
}

// weight를 콘솔에 출력한다.
void Perceptron::print_weights() {
	cout << "weights: ";
	for (const double &w : weights) {
		cout << w << " ";
	}
	cout << "\n";
}

// weight에 따른 직선의 방정식을 출력한다.
// 2차원 입력에만 유효
void Perceptron::print_linear_function() {
	// 입력 차원 체크
	if (input_dim != 2)
		return;

	// 직선의 방정식 출력
	cout << "직선의 방정식: x2 = ";
	cout << (weights[0] / weights[1] > 0 ? "-" : "") << abs(weights[0] / weights[1]) << " * x1";
	cout << (weights[2] / weights[1] > 0 ? " - " : " + ") << abs(weights[2] / weights[1]) << "\n";
}

// 직선의 방정식을 파일로 저장한다.
// 2차원 입력에만 유효
// outFile: 직선의 방정식을 출력할 파일
void Perceptron::write_linear_function(ofstream &outFile) {
	// 입력 차원 체크
	if (input_dim != 2)
		return;

	// 직선의 방정식 저장
	outFile << (-1 * weights[0] / weights[1]) << " " << (-1 * weights[2] / weights[1]) << "\n";
}

// forward 연산
// x: input값을 저장한 vector
// 반환값: forward 연산 결과
double Perceptron::forward(const vector<double> &x) {
	// 올바른 입력인지 체크
	if (x.size() != input_dim) {
		cout << "Perceptron error: x.size() != input_dim.\n";
		exit(-1);
	}

	// forward 연산 시작
	double result = 0;

	// 각 input에 weight를 곱한 값의 합을 구한다.
	for (int i = 0; i < input_dim; i++)
		result += x[i] * weights[i];
	result += weights[input_dim];	// Threashold 값

	// net 값을 저장해둔다. (이후 delta 값 구하는 데 활용)
	net = result;

	// activation function
	result = activate(result);

	return result;
}

// weight update
// x: input값을 저장한 vector, delta_bar: 이전 layer에서 넘어온 delta_bar
// 반환값: 다음 layer에 넘길 delta_bar를 계산하기 위한 값이 담긴 vector (w * delta)
vector<double> Perceptron::update_weight(const vector<double> &x, const double &delta_bar) {
	// delta 계산
	delta = delta_bar * SigmoidPrime(net);

	// 다음 layer에 넘겨줄 정보 계산 (w * delta)
	vector<double> rtn;
	for (int i = 0; i < input_dim; i++)
		rtn.push_back(weights[i] * delta);

	// weight update
	for (int i = 0; i < input_dim; i++)
		weights[i] += (-learning_rate * delta * x[i]);

	// threshold update
	weights[input_dim] += (-learning_rate * delta);

	return rtn;
}

// weight를 랜덤한 값으로 초기화한다.
void Perceptron::weight_random_initialize() {
	// random number generator 정의
	random_device rd;
	mt19937 e2(rd());
	uniform_real_distribution<> dist(-1, 1);

	// random number로 weight 초기화
	weights.resize(input_dim + 1);		// 외부엔 보이지 않는 weight 하나 추가 (threshold)
	for (int i = 0; i < input_dim + 1; i++)
		weights[i] = dist(e2);
}

// 활성화 함수
double Perceptron::activate(double in) {
	return Sigmoid(in);		// Sigmoid 함수 적용
//	return hard_limiting(in);
}
