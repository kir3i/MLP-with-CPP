#include <iostream>
#include <fstream>
#include <vector>
#include "Perceptron.h"
#include "Layer.h"

using namespace std;

// 생성자
// input_dim: layer의 입력 차원, output_dim: layer의 출력 차원
// lr: learning rate
Layer::Layer(int input_dim, int output_dim, double lr) {
	this->input_dim = input_dim;			// 입력 차원 초기화
	this->output_dim = output_dim;			// 출력 차원 초기화

	// layer를 구성할 perceptron 생성
	for (int i = 0; i < output_dim; i++)
		nodes.push_back(Perceptron(input_dim, lr));
}

// Layer에 속하는 node들의 weight를 file에 기록
// outFile: weight를 저장할 file
// 저장형식
// - Layer 별로 하나의 행렬을 생성한다.
// - 각 row는 한 노드의 weight를 나타낸다. bias는 제일 마지막에 기록된다.
void Layer::write_weight(ofstream &outFile) {
	for (Perceptron &n : nodes) {
		for (const double &w : n.get_weights())
			outFile << w << " ";
		outFile << "\n";
	}
}

// layer에 속한 nodes들의 직선의 방정식을 콘솔에 출력한다.
// 표시형식: [레이어 번호]-[노드 번호] 직선의 방정식: [직선의 방정식]
// layer_num: layer의 번호 (몇 번째 layer인지)
void Layer::print_linear_function(const int &layer_num) {
	for (int i = 0; i < nodes.size(); i++) {
		cout << layer_num << "-" << i + 1 << " ";
		nodes[i].print_linear_function();
	}
}

// layer에 속한 nodes들의 직선의 방정식을 file에 저장한다.
// outFile: 직선의 방정식을 저장할 파일
void Layer::write_linear_function(ofstream &outFile) {
	for (Perceptron &n : nodes)
		n.write_linear_function(outFile);
	outFile << "\n";
}

// 직전 forward 연산에서 사용했던 input을 반환한다.
vector<double> Layer::get_prev_x() {
	return prev_x;
}

// foward 연산
// x: layer에 들어온 입력
// 반환값: forward 연산 결과
vector<double> Layer::forward(const vector<double> &x) {
	// 올바른 입력인지 체크
	if (x.size() != input_dim) {
		cout << "Layer error: x.size() != input_dim\n";
		cout << "input.size() == " << x.size() << "\n";
		cout << "input_dim == " << input_dim << "\n";
		exit(-1);
	}

	prev_x = x;			// 입력을 저장해둔다. (backward에서 사용)

	// forward 연산 시작
	vector<double> rtn;
	// layer에 속한 perceptron 별로 forward 연산
	for (Perceptron &p : nodes)
		rtn.push_back(p.forward(x));

	return rtn;
}

// backward 연산
// delta_bar: 현재 layer 기준 노드 별 delta_bar vector
// 반환값: 다음 layer를 위한 delta_bar vector
vector<double> Layer::backward(const vector<double> &delta_bar) {
	// 올바른 입력인지 체크
	if (delta_bar.size() != output_dim) {
		cout << "Layer error: delta_bar.size() != output_dim\n";
		exit(-1);
	}

	// backward 연산 시작
	vector<double> rtn(input_dim, 0);		// 다음 layer를 위한 delta_bar vector

	// layer에 속한 perceptron에 대해 delta_bar를 이용해 weight update
	for (int i = 0; i < output_dim; i++) {
		vector<double> update_delta = nodes[i].update_weight(prev_x, delta_bar[i]);

		// 다음 layer를 위한 delta_bar 계산
		for (int j = 0; j < input_dim; j++)
			rtn[j] += update_delta[j];
	}

	return rtn;
}
