// 컴퓨터과학부 2016920029 유시온
// 인공지능 과제 #2 입력이 n차원인 perceptron learning 구현
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>

using namespace std;

// Hard Limiting
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
	// 생성자
	Perceptron(int n, double lr) {
		this->input_dim = n;				// 입력 차원 초기화
		this->weight_random_initialize();	// weight 초기화
		this->learning_rate = lr;			// learning_rate 초기화
	}		

	// weight를 초기화한다.
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

	// weight를 출력한다.
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
	// 반환값: 연산 결과
	double forward(const vector<double> &x) {
		// 올바른 입력인지 체크
		if (x.size() != input_dim) {
			cout << "Perceptron의 input_dim과 input_vals.size()가 일치하지 않습니다.\n";
			exit(-1);
		}

		// forward 연산 시작
		double result = 0;
		// 각 input에 weight를 곱한 값의 합을 구한다.
		for (int i = 0; i < input_dim; i++)
			result += x[i] * weights[i];
		result += weights[input_dim];	// Threashold 값

		net = result;

		// activation function
		result = activate(result);
		
		return result;
	}

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

	vector<double> update_weight(const vector<double> &x, const double &delta_bar) {
		delta = delta_bar * SigmoidPrime(net);
		
		// 다음 레이어에 넘겨줄 정보 계산
		vector<double> rtn;
		for (int i = 0; i < input_dim; i++)
			rtn.push_back(weights[i] * delta);

		// weight 업데이트
		for (int i = 0; i < input_dim; i++)
			weights[i] += (-learning_rate * delta * x[i]);
		
		// threshold 업데이트
		weights[input_dim] += (-learning_rate * delta);

		return rtn;
	}

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

private:
	int input_dim = 0;				// input 차원
	double learning_rate;			// learning_rate
	vector<double> weights;			// perceptron의 weight 값
	double net;
	double delta;

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
		//return Sigmoid(in);
		return hard_limiting(in);
	}

};

class Layer {
public:
	Layer(int input_dim, int output_dim, double lr) {
		this->input_dim = input_dim;
		this->output_dim = output_dim;
		this->learning_rate = lr;
		for (int i = 0; i < output_dim; i++)
			nodes.push_back(Perceptron(input_dim, lr));
	}
	
	vector<double> forward(const vector<double> &x) {
		if (x.size() != input_dim) {
			cout << "Layer error: x.size() != input_dim\n";
			cout << "input.size() == " << x.size() << "\n";
			cout << "input_dim == " << input_dim << "\n";
			exit(-1);
		}

		vector<double> rtn;
		for (Perceptron &p : nodes)
			rtn.push_back(p.forward(x));
		return rtn;
	}

	// delta_bar: 현재 레이어 기준 노드 별 delta_bar 값
	vector<double> backward(const vector<double> &x, const vector<double> &delta_bar) {
		if (delta_bar.size() != output_dim) {
			cout << "Layer error: delta_bar.size() != output_dim\n";
			exit(-1);
		}

		// 다음 레이어를 위한 delta_bar
		vector<double> rtn(input_dim, 0);

		for (int i = 0; i < output_dim; i++) {
			vector<double> update_delta = nodes[i].update_weight(x, delta_bar[i]);

			// 다음 레이어를 위한 delta_bar 업데이트
			for (int j=0; j<input_dim; j++)
				rtn[j] += update_delta[j];
		}

		return rtn;
	}

	double run(const vector<vector<double>> &input, const vector<double> &target) {
		// 올바른 입력인지 체크
		if (input.size() != target.size()) {
			cout << "input과 target의 개수가 일치하지 않습니다.\n";
			exit(-1);
		}

		for (int i = 0; i < input.size(); i++) {
			forward(input[i]);
		}
	}

private:
	vector<Perceptron> nodes;
	int input_dim;
	int output_dim;
	double learning_rate;
};

class Model {
public:
	// input_dim: 입력 data의 차원
	// layers_dim: 각 layer의 output_dim을 갖는 벡터
	// lr: learning rate
	Model(int input_dim, const vector<int> &layers_dim, double lr) {
		if (layers_dim.size() == 0) {
			cout << "Model error: layers_dim.size() == 0\n";
			exit(-1);
		}

		this->learning_rate = lr;
		layers.push_back(Layer(input_dim, layers_dim[0], lr));
		for (int i = 1; i < layers_dim.size(); i++)
			layers.push_back(Layer(layers_dim[i - 1], layers_dim[i], lr));
	}

	vector<double> forward(const vector<double> &x) {
		vector<double> rtn;
		vector<double> nx = x;
		for (Layer &l : layers) {
			rtn.clear();
			rtn = l.forward(nx);
			nx = rtn;
		}

		return rtn;
	}

	void backward(const vector<double> &x, const vector<double> &y, const vector<double> &target) {
		// 태초의 delta_bar
		vector<double> delta_bar;
		for (int i = 0; i < y.size(); i++)
			delta_bar.push_back(-target[i] + y[i]);

		// back propagation
		for (int i = (int)layers.size() - 1; i >= 0; i--) {
			delta_bar = layers[i].backward(x, delta_bar);
		}
	}

	vector<double> run(const vector<double> &x, const vector<double> &target) {
		
		vector<double> y = forward(x);

		if (y.size() != target.size()) {
			cout << "Model error: y.size() != target.size()\n";
			exit(-1);
		}

		backward(x, y, target);

		return y;
	}

	double run(const vector<vector<double>> &input, const vector<vector<double>> &target) {
		if (input.size() != target.size()) {
			cout << "Model error: input.size() != target.size()\n";
			exit(-1);
		}

		double loss = 0;

		for (int i = 0; i < input.size(); i++) {
			vector<double> predict = run(input[i], target[i]);
			loss += mean_squared_error(predict, target[i]);

			//test
			for (const auto &x : input[i])
				cout << x << " ";
			cout << " -> " << predict[0] << "\n";
		}

		return loss;
	}

private:
	vector<Layer> layers;
	double learning_rate;
};

// main 함수
int main(void) {
	const int AND = 1;
	const int OR = 2;
	const int XOR = 3;
	const int DONUT = 4;

	// 입력 차원을 2로 고정한다.
//	int N = 2;	//cin >> N;
//	if (N <= 0)	return 0;

	// learning rate 입력
	double lr;	
	cout << "learning rate를 입력하세요: ";		cin >> lr;

	// 테스트할 gate 선택
	int select = 0;
	cout << "테스트할 gate를 고르세요\n";
	cout << "1: AND, 2: OR, 3: XOR\n";
	cout << "입력: ";	cin >> select;
	string filename;




	//테스트할 gate에 따른 input, target 설정
	vector<vector<double>> x;
//	vector<double> target;
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
		cout << "Gate를 잘못 선택했습니다.\n";
		exit(-1);
	}


	vector<int> layers_dim = { 2, 1 };
	Model m = Model(x[0].size(), layers_dim, lr);

	// 테스트 수행
	cout << "\n====================실행결과====================\n";
	double loss = 1;
	for (int epoch = 1; loss > 0.01; epoch++) {
		cout << "epoch: " << epoch << "\n";
		loss = m.run(x, target);
		cout << "loss: " << loss << "\n\n";	// loss 출력
	}
	

/*

	// Perceptron 생성
	Perceptron p = Perceptron(x[0].size(), lr);
//	ofstream lineFile(filename + ".txt");
//	ofstream lossFile(filename + "_loss.txt");
	double loss = 1;
	for (int epoch = 1; loss > 0.1; epoch++) {
		cout << "epoch: " << epoch << "\n";
		p.print_weights();					// weight 출력
		p.print_linear_function();			// 직선의 방정식 출력
		//p.write_linear_function(lineFile);	// 직선의 방정식 저장
		loss = p.run(x, target);			// perceptron 연산
		cout << "loss: " << loss << "\n\n";	// loss 출력
		//lossFile << loss << "\n";			// loss 저장
			
		// epoch가 10만회를 넘으면 정지
		if (epoch >= 100000)
			break;
	}
//	lineFile.close();
//	lossFile.close();
	return 0;
*/
}