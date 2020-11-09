#pragma once

using namespace std;

// Model class
class Model {
public:
	// 생성자
	// input_dim: 입력 data의 차원, layers_dim: 각 layer의 output_dim을 갖는 벡터
	// lr: learning rate
	Model(int input_dim, const vector<int> &layers_dim, double lr);

	// Model의 각 Layer의 node들의 weight를 file에 기록한다.
	// outFile: weight가 저장될 file
	// 저장형식
	// - Layer 별로 하나의 행렬을 생성한다.
	// - 각 row는 한 노드의 weight를 나타낸다. bias는 제일 마지막 요소이다.
	void write_weights(ofstream &outFile);

	// model에 속한 layer들에 속한 각 node의 직선의 방정식을 콘솔에 출력한다.
	// 출력형식: [layer 번호]-[node번호] 직선의 방정식: [직선의 방정식]
	// layer번호와 node 번호 모두 1번부터 시작한다.
	void print_linear_function();

	// model에 속한 layer들에 속한 각 node의 직선의 방정식을 file에 저장한다.
	// 각 epoch는 ","로 구분한다.
	// outfile: 직선의 방정식을 저장할 file
	void write_linear_function(ofstream &outFile);

	// 각 입력의 hidden layer를 통한 이동을 file로 출력한다.
	// {2, 1}에 해당하는 모델에만 정상적으로 작동한다.
	// x: model에 들어온 입력, outFile: 기록할 파일
	// 각 epoch는 "\n\n"으로 구분한다.
	// 위에서부터 (0, 0), (0, 1), (1, 0), (1, 1)이 이동한 점을 출력한다.
	void write_dot_moving(const vector<double> &x, ofstream &outFile);

	// forward 연산
	// x: model에 들어온 입력
	// 반환값: forward 연산 결과
	vector<double> forward(const vector<double> &x);

	// backward 연산
	// x: 입력 data, y: model로 예측한 결과, target: 실제 결과
	void backward(const vector<double> &x, const vector<double> &y, const vector<double> &target);

	// model 작동, 한 case에 대해서 학습
	// x: model에 들어온 input, target: 정답 결과
	// 반환값: model의 연산 결과 (예측 결과)
	vector<double> run(const vector<double> &x, const vector<double> &target);

	// model 작동, 다수의 case에 대해서 학습
	// x: model에 들어온 input, target: 정답 결과
	// outFile: dot_moving을 출력할 파일
	// print: 해당 epoch에서 결과값 출력, dot_moving: 점 이동 결과 출력 ({2, 1} 모델에서만 작동)
	// 반환값: 예측한 결과의 loss값
	double run(const vector<vector<double>> &input, const vector<vector<double>> &target, ofstream &outFile, const bool &print = true, const bool &dot_moving = true);

private:
	vector<Layer> layers;	// model을 구성하는 layer vector
};