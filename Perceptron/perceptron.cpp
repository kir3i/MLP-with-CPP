// 컴퓨터과학부 2016920029 유시온
// 인공지능 과제 #3 Multi-Layer perceptron 구현
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>

using namespace std;

// Hard Limiting (기준값: 0)
double hard_limiting(double x) {
	if (x <= 0)
		return 0;
	else
		return 1;
}

// Sigmoid 
double Sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

// Sigmoid 도함수
double SigmoidPrime(double x) {
	return Sigmoid(x) * (1 - Sigmoid(x));
}

// Mean Squared Error
double mean_squared_error(vector<double> y, vector<double> target) {
	if (y.size() != target.size()) {
		cout << "mse error: y.size() != target.size()\n";
		exit(-1);
	}
	double rtn = 0;
	for (int i = 0; i < y.size(); i++)
		rtn += (target[i] - y[i]) * (target[i] - y[i]);
	return rtn / 2;
}

// Perceptron class
class Perceptron {
public:
	// Perceptron 생성자
	// n: 입력 차원, lr: learning rate
	Perceptron(int n, double lr) {
		this->input_dim = n;				// 입력 차원 초기화
		this->weight_random_initialize();	// weight 초기화
		this->learning_rate = lr;			// learning_rate 초기화
	}		

	// weight를 콘솔에 출력한다.
	void print_weights() {
		cout << "weights: ";
		for (const double &w : weights) {
			cout << w << " ";
		}
		cout << "\n";
	}

	// weight에 따른 직선의 방정식을 출력한다.
	// (2차원 입력에만 유효함)
	void print_linear_function() {
		// 입력 차원 체크
		if (input_dim != 2)
			return;

		// 직선의 방정식 출력
		cout << "직선의 방정식: x2 = ";
		cout << (weights[0] / weights[1] > 0 ? "-" : "") << abs(weights[0] / weights[1]) << " * x1";
		cout << (weights[2] / weights[1] > 0 ? " - " : " + ") << abs(weights[2] / weights[1]) << "\n";
	}

	// 직선의 방정식을 파일로 저장한다.
	void write_linear_function(ofstream &outFile) {
		// 입력 차원 체크
		if (input_dim != 2)
			return;

		// 직선의 방정식 저장
		outFile << (-1 * weights[0] / weights[1]) << " " << (-1 * weights[2] / weights[1]) << "\n";
	}

	// forward 연산
	// x: input값을 저장한 vector
	// 반환값: forward 연산 결과
	double forward(const vector<double> &x) {
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

/*
	// weight update(learning), 한 개의 case에 대해 업데이트한다.
	// back propagation을 통해 weight를 갱신한다.
	// x: input값을 저장한 vector, y: 결과값, target: 올바른 결과값
	void update_weight(const vector<double> &x, const double &y, const double &target) {
		for (int i = 0; i < input_dim; i++)
			weights[i] += learning_rate * (target - y) * x[i];
		weights[input_dim] += learning_rate * (target - y);
	}

	// weight update(learning), 여러 개의 case에 대해 업데이트한다.
	// back propagation을 통해 weight를 갱신한다.
	// x: input값을 저장한 vector, y: 결과값, target: 올바른 결과값
	void update_weight(const vector<vector<double>> &input, const vector<double> &y, const vector<double> &target) {
		for (int i = 0; i < input.size(); i++)
			update_weight(input[i], y[i], target[i]);
	}
*/
	
	// weight update
	// x: input값을 저장한 vector, delta_bar: 이전 layer에서 넘어온 delta_bar
	// 반환값: 다음 layer에 넘길 delta_bar를 계산하기 위한 값이 담긴 vector (w * delta)
	vector<double> update_weight(const vector<double> &x, const double &delta_bar) {
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
/*
	// Perceptron 작동 (연산)
	// input: input값을 저장한 vector, target: 올바른 결과값
	// 반환값: Perceptron 연산 결과에 따른 loss 값
	double run(const vector<vector<double>> &input, const vector<double> &target) {
		// 올바른 입력인지 체크
		if (input.size() != target.size()) {
			cout << "input과 target의 개수가 일치하지 않습니다.\n";
			exit(-1);
		}

		// Perceptron 연산 시작
		double loss = 0;
		vector<double> y(input.size());
		// case마다 연산
		for (int i = 0; i < input.size(); i++) {
			y[i] = forward(input[i]);					// forward 연산
			loss += (target[i] - y[i]) * (target[i] - y[i]) / 2;	// mean squared error
			//update_weight(input[i], y[i], target[i]);
		}
		update_weight(input, y, target);							// backward 연산
		return loss;
	}
*/

private:
	int input_dim = 0;				// input 차원
	double learning_rate;			// learning_rate
	vector<double> weights;			// perceptron의 weight 값
	double net;						// net값 (activation 함수 들어가기 전)
	double delta;					// delta값, 이전 layer에서 넘어온 delta_bar로 계산
/*
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
*/
	
	// weight를 랜덤한 값으로 초기화한다.
	void weight_random_initialize() {
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
	double activate(double in) {
		return Sigmoid(in);	// Sigmoid 함수 적용
	//	return hard_limiting(in);
	}
};

// Layer Class
class Layer {
public:
	// 생성자
	// input_dim: layer의 입력 차원, output_dim: layer의 출력 차원
	// lr: learning rate
	Layer(int input_dim, int output_dim, double lr) {
		this->input_dim = input_dim;			// 입력 차원 초기화
		this->output_dim = output_dim;			// 출력 차원 초기화
		
		// layer를 구성할 perceptron 생성
		for (int i = 0; i < output_dim; i++)
			nodes.push_back(Perceptron(input_dim, lr));
	}
	
	// foward 연산
	// x: layer에 들어온 입력
	// 반환값: forward 연산 결과
	vector<double> forward(const vector<double> &x) {
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
	vector<double> backward(const vector<double> &delta_bar) {
		// 올바른 입력인지 체크
		if (delta_bar.size() != output_dim) {
			cout << "Layer error: delta_bar.size() != output_dim\n";
			exit(-1);
		}

		// backward 연산 시작
		vector<double> rtn(input_dim, 0);		// 다음 layer를 위한 delta_bar vector
		
		for (int i = 0; i < output_dim; i++) {
			// layer에 속한 perceptron에 대해 delta_bar를 이용해 weight update
			vector<double> update_delta = nodes[i].update_weight(prev_x, delta_bar[i]);

			// 다음 layer를 위한 delta_bar 계산
			for (int j = 0; j < input_dim; j++)
				rtn[j] += update_delta[j];
		}

		return rtn;
	}

private:
	vector<Perceptron> nodes;	// layer를 구성하는 perceptron vector
	int input_dim;				// 입력 차원
	int output_dim;				// 출력 차원
	vector<double> prev_x;		// forward 연산에 사용한 입력
};

// Model class
class Model {
public:
	// 생성자
	// input_dim: 입력 data의 차원, layers_dim: 각 layer의 output_dim을 갖는 벡터
	// lr: learning rate
	Model(int input_dim, const vector<int> &layers_dim, double lr) {
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

	// forward 연산
	// x: model에 들어온 입력
	// 반환값: forward 연산 결과
	vector<double> forward(const vector<double> &x) {
		// layer 순서대로 통과시키며 연산
		vector<double> nx = x;
		for (Layer &l : layers)
			nx = l.forward(nx);

		return nx;
	}

	// backward 연산
	// x: 입력 data, y: model로 예측한 결과, target: 실제 결과
	void backward(const vector<double> &x, const vector<double> &y, const vector<double> &target) {
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
	vector<double> run(const vector<double> &x, const vector<double> &target) {
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
	// 반환값: 예측한 결과의 loss값
	double run(const vector<vector<double>> &input, const vector<vector<double>> &target) {
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

			// 예측 결과 출력
			for (const double &x : input[i])
				cout << x << " ";
			cout << " -> ";
			for (const double &y: predict)
				cout << y << " ";
			cout << "\n";
		}

		return loss;
	}

private:
	vector<Layer> layers;	// model을 구성하는 layer vector
};

// main 함수
int main(void) {
	const double TOLERANCE = 0.01;
	const int AND = 1;
	const int OR = 2;
	const int XOR = 3;
	const int DONUT = 4;

	// 테스트할 data 선택
	int select = 0;
	cout << "테스트할 data를 고르세요\n";
	cout << "1: AND, 2: OR, 3: XOR 4: DONUT\n";
	cout << "입력: ";	cin >> select;
	string filename;

	//테스트할 data에 따른 input, target 설정
	vector<vector<double>> x;
	vector<vector<double>> target;
	switch (select) {
	case AND:
		x = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
		target = { {0}, {0}, {0}, {1} };
		filename = "AND";
		break;
	case OR:
		x = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
		target = { {0}, {1}, {1}, {1} };
		filename = "OR";
		break;
	case XOR:
		x = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
		target = { {0}, {1}, {1}, {0} };
		filename = "XOR";
		break;
	case DONUT:
		x = { {0, 0}, {0, 1}, {1, 0}, {1, 1}, {0.5, 1}, {1, 0.5}, {0, 0.5}, {0.5, 0}, {0.5, 0.5} };
		target = { {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {1} };
		filename = "DONUT";
		break;
	default:
		cout << "Data를 잘못 선택했습니다.\n";
		exit(-1);
	}


	// layer 수 설정
	int layer_num = 0;
	cout << "layer 수를 정하세요: ";	cin >> layer_num;
	
	// layer 별로 출력 차원 설정
	cout << "각 layer의 output 차원을 띄어쓰기로 구분하여 입력하세요: ";
	vector<int> layers_dim(layer_num);
	for (int i = 0; i < layer_num; i++)
		cin >> layers_dim[i];

	// 올바른 입력인지 체크
	if (layers_dim[layer_num - 1] != target[0].size()) {
		cout << "최종 출력 차원이 target과 일치하지 않습니다.\n";
		exit(-1);
	}

	// learning rate 입력
	double lr;
	cout << "learning rate를 입력하세요: ";		cin >> lr;

	// model 생성
	Model m = Model(x[0].size(), layers_dim, lr);

	// 테스트 수행
	cout << "\n====================실행결과====================\n";
	double loss = 1;
	for (int epoch = 1; loss > TOLERANCE; epoch++) {
		cout << "epoch: " << epoch << "\n";
		loss = m.run(x, target);
		cout << "loss: " << loss << "\n\n";	// loss 출력
	}

	return 0;
	
	// test 결과
	// 1. {4, 1} + lr = 0.5로 했을 때 모두 성공함. 도넛은 약 5천회
	// 2. {2, 4, 4, 1} + lr = 0.7로 하면 잘 안 끝남.
	// TODO
	// 1. 데이터 분석 (노드마다 직선 그래프, epoch마다 error 그래프)
	// 2. weight는 행렬 형식으로 파일에 저장(?)
	// 3. 결과보고서 작성
}