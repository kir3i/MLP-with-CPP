#pragma once

using namespace std;

// Perceptron class
class Perceptron {
public:
	// Perceptron 생성자
	// n: 입력 차원, lr: learning rate
	Perceptron(int n, double lr);

	// weight를 반환한다.
	vector<double> get_weights();

	// weight를 콘솔에 출력한다.
	void print_weights();

	// weight에 따른 직선의 방정식을 출력한다.
	// 2차원 입력에만 유효
	void print_linear_function();

	// 직선의 방정식을 파일로 저장한다.
	// 2차원 입력에만 유효
	// outFile: 직선의 방정식을 출력할 파일
	void write_linear_function(ofstream &outFile);

	// forward 연산
	// x: input값을 저장한 vector
	// 반환값: forward 연산 결과
	double forward(const vector<double> &x);

	// weight update
	// x: input값을 저장한 vector, delta_bar: 이전 layer에서 넘어온 delta_bar
	// 반환값: 다음 layer에 넘길 delta_bar를 계산하기 위한 값이 담긴 vector (w * delta)
	vector<double> update_weight(const vector<double> &x, const double &delta_bar);

private:
	int input_dim = 0;				// input 차원
	double learning_rate;			// learning_rate
	vector<double> weights;			// perceptron의 weight 값
	double net;						// net값 (activation 함수 들어가기 전)
	double delta;					// delta값, 이전 layer에서 넘어온 delta_bar로 계산

	// weight를 랜덤한 값으로 초기화한다.
	void weight_random_initialize();

	// 활성화 함수
	double activate(double in);
};