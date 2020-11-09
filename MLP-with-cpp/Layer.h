#pragma once
#include "Perceptron.h"

using namespace std;

// Layer Class
class Layer {
public:
	// 생성자
	// input_dim: layer의 입력 차원, output_dim: layer의 출력 차원
	// lr: learning rate
	Layer(int input_dim, int output_dim, double lr);

	// Layer에 속하는 node들의 weight를 file에 기록
	// outFile: weight를 저장할 file
	// 저장형식
	// - Layer 별로 하나의 행렬을 생성한다.
	// - 각 row는 한 노드의 weight를 나타낸다. bias는 제일 마지막에 기록된다.
	void write_weight(ofstream &outFile);

	// layer에 속한 nodes들의 직선의 방정식을 콘솔에 출력한다.
	// 표시형식: [레이어 번호]-[노드 번호] 직선의 방정식: [직선의 방정식]
	// layer_num: layer의 번호 (몇 번째 layer인지)
	void print_linear_function(const int &layer_num);

	// layer에 속한 nodes들의 직선의 방정식을 file에 저장한다.
	// outFile: 직선의 방정식을 저장할 파일
	void write_linear_function(ofstream &outFile);

	// 직전 forward 연산에서 사용했던 input을 반환한다.
	vector<double> get_prev_x();

	// foward 연산
	// x: layer에 들어온 입력
	// 반환값: forward 연산 결과
	vector<double> forward(const vector<double> &x);

	// backward 연산
	// delta_bar: 현재 layer 기준 노드 별 delta_bar vector
	// 반환값: 다음 layer를 위한 delta_bar vector
	vector<double> backward(const vector<double> &delta_bar);

private:
	vector<Perceptron> nodes;	// layer를 구성하는 perceptron vector
	int input_dim;				// 입력 차원
	int output_dim;				// 출력 차원
	vector<double> prev_x;		// forward 연산에 사용한 입력
};