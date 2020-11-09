#include <vector>
#include <iostream>
#include <fstream>
#include "objective_functions.h"
#include "Layer.h"
#include "Model.h"

using namespace std;

// 생성자
// input_dim: 입력 data의 차원, layers_dim: 각 layer의 output_dim을 갖는 벡터
// lr: learning rate
Model::Model(int input_dim, const vector<int> &layers_dim, double lr) {
	// 올바른 입력인지 체크
	if (layers_dim.size() == 0) {
		cout << "Model error: layers_dim.size() == 0\n";
		exit(-1);
	}

	// model 구성
	layers.push_back(Layer(input_dim, layers_dim[0], lr));
	for (int i = 1; i < layers_dim.size(); i++)
		layers.push_back(Layer(layers_dim[i - 1], layers_dim[i], lr));
}

// Model의 각 Layer의 node들의 weight를 file에 기록한다.
// outFile: weight가 저장될 file
// 저장형식
// - Layer 별로 하나의 행렬을 생성한다.
// - 각 row는 한 노드의 weight를 나타낸다. bias는 제일 마지막 요소이다.
void Model::write_weights(ofstream &outFile) {
	for (int i = 0; i < layers.size(); i++) {
		outFile << "Layer " << i + 1 << "\n";
		layers[i].write_weight(outFile);
		outFile << "\n";
	}
}

// model에 속한 layer들에 속한 각 node의 직선의 방정식을 콘솔에 출력한다.
// 출력형식: [layer 번호]-[node번호] 직선의 방정식: [직선의 방정식]
// layer번호와 node 번호 모두 1번부터 시작한다.
void Model::print_linear_function() {
	for (int i = 0; i < layers.size(); i++)
		layers[i].print_linear_function(i);
}

// model에 속한 layer들에 속한 각 node의 직선의 방정식을 file에 저장한다.
// 각 epoch는 ","로 구분한다.
// outfile: 직선의 방정식을 저장할 file
void Model::write_linear_function(ofstream &outFile) {
	for (Layer &l : layers)
		l.write_linear_function(outFile);
	outFile << ",";
}

// 각 입력의 hidden layer를 통한 이동을 file로 출력한다.
// {2, 1}에 해당하는 모델에만 정상적으로 작동한다.
// x: model에 들어온 입력, outFile: 기록할 파일
// 각 epoch는 "\n\n"으로 구분한다.
// 위에서부터 (0, 0), (0, 1), (1, 0), (1, 1)이 이동한 점을 출력한다.
void Model::write_dot_moving(const vector<double> &x, ofstream &outFile) {
	// 올바르지 않은 호출
	if (layers.size() != 2 || x.size() != 2)
		return;

	// 두 번째 레이어의 입력을 체크한다.
	vector<double> in = layers[1].get_prev_x();
	if (in.size() != 2) {
		cout << "model error: write_dot_moving은 이차원 입력에만 작동하는 함수입니다.\n";
		return;
	}
	outFile << in[0] << " " << in[1] << "\n";
}

// forward 연산
// x: model에 들어온 입력
// 반환값: forward 연산 결과
vector<double> Model::forward(const vector<double> &x) {
	// layer 순서대로 통과시키며 연산
	vector<double> nx = x;
	for (Layer &l : layers)
		nx = l.forward(nx);

	return nx;
}

// backward 연산
// x: 입력 data, y: model로 예측한 결과, target: 실제 결과
void Model::backward(const vector<double> &x, const vector<double> &y, const vector<double> &target) {
	// 초기 delta_bar 게산
	vector<double> delta_bar;
	for (int i = 0; i < y.size(); i++)
		delta_bar.push_back(-target[i] + y[i]);

	// back propagation 수행
	for (int i = (int)layers.size() - 1; i >= 0; i--)
		delta_bar = layers[i].backward(delta_bar);		// delta_bar 업데이트
}

// model 작동, 한 case에 대해서 학습
// x: model에 들어온 input, target: 정답 결과
// 반환값: model의 연산 결과 (예측 결과)
vector<double> Model::run(const vector<double> &x, const vector<double> &target) {
	// forward 연산 수행
	vector<double> y = forward(x);

	// 올바른 입력인지 체크
	if (y.size() != target.size()) {
		cout << "Model error: y.size() != target.size()\n";
		exit(-1);
	}

	// backward 연산 수행
	backward(x, y, target);

	return y;
}

// model 작동, 다수의 case에 대해서 학습
// x: model에 들어온 input, target: 정답 결과
// outFile: dot_moving을 출력할 파일
// print: 해당 epoch에서 결과값 출력, dot_moving: 점 이동 결과 출력 ({2, 1} 모델에서만 작동)
// 반환값: 예측한 결과의 loss값
double Model::run(const vector<vector<double>> &input, const vector<vector<double>> &target, ofstream &outFile, const bool &print, const bool &dot_moving) {
	// 올바른 입력인지 체크
	if (input.size() != target.size()) {
		cout << "Model error: input.size() != target.size()\n";
		exit(-1);
	}

	// model 작동 시작
	double loss = 0;

	// 케이스 별로 계산
	for (int i = 0; i < input.size(); i++) {
		vector<double> predict = run(input[i], target[i]);	// 예측값 계산
		loss += mean_squared_error(predict, target[i]);		// loss 계산

		// 점 이동 결과 출력
		if (dot_moving)
			write_dot_moving(input[i], outFile);

		// 예측 결과 출력
		if (print) {
			for (const double &x : input[i])
				cout << x << " ";
			cout << " -> ";
			for (const double &y : predict)
				cout << y << " ";
			cout << "\n";
		}
	}
	outFile << "\n";

	// loss 리턴
	return loss;
}