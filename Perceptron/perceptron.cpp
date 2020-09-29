// 컴퓨터과학부 2016920029 유시온
// 인공지능 과제 #2 입력이 n차원인 perceptron learning 구현
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <string>

using namespace std;

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

	// 활성화 함수
	double activate(double in) {
		// Hard Limiting
		if (in <= THRESHOLD)
			return 0;
		else
			return 1;
	}

	// forward 연산
	// x: input값을 저장한 vector
	// 반환값: 연산 결과
	double foward(const vector<double> &x) {
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
		// activation function
		result = activate(result);
		
		return result;
	}

	// weight update(learning), 한 개의 case에 대해 업데이트한다.
	// back propagation을 통해 weight를 갱신한다.
	// x: input값을 저장한 vector, y: 결과값, target: 올바른 결과값
	void update_weight(const vector<double> &x, const double &y, const double &target) {
		for (int i = 0; i < input_dim; i++)
			weights[i] += learning_rate * (target - y)*x[i];
		weights[input_dim] += learning_rate * (target - y);
	}

	// weight update(learning), 여러 개의 case에 대해 업데이트한다.
	// back propagation을 통해 weight를 갱신한다.
	// x: input값을 저장한 vector, y: 결과값, target: 올바른 결과값
	void update_weight(const vector<vector<double>> &input, const vector<double> &y, const vector<double> &target) {
		for (int i = 0; i < input.size(); i++)
			update_weight(input[i], y[i], target[i]);
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
			y[i] = foward(input[i]);								// forward 연산
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
	const static int THRESHOLD = 0;
};

// main 함수
int main(void) {
	const int AND = 1;
	const int OR = 2;
	const int XOR = 3;

	// 입력 차원을 2로 고정한다.
	int N = 2;	//cin >> N;
	if (N <= 0)	return 0;

	// learning rate 입력
	double lr;	
	cout << "learning rate를 입력하세요: ";		cin >> lr;

	// 테스트할 gate 선택
	int select = 0;
	cout << "테스트할 gate를 고르세요\n";
	cout << "1: AND, 2: OR, 3: XOR\n";
	cout << "입력: ";	cin >> select;
	string filename;

	// Perceptron 생성
//	Perceptron p = Perceptron(N, lr);
	

	//테스트할 gate에 따른 input, target 설정
	vector<vector<double>> x = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
	vector<double> target;
	switch (select) {
	case AND:
		target = { {0}, {0}, {0}, {1} };
		filename = "AND";
		break;
	case OR:
		target = { {0}, {1}, {1}, {1} };
		filename = "OR";
		break;
	case XOR:
		target = { {0}, {1}, {1}, {0} };
		filename = "XOR";
		break;
	default:
		cout << "Gate를 잘못 선택했습니다.\n";
		exit(-1);
	}

	// 테스트 수행
	cout << "\n====================실행결과====================\n";
	
	ofstream epochFile(filename + "_lr=" + to_string(lr) + "_epoch.txt");
//	ofstream lineFile(filename + ".txt");
//	ofstream lossFile(filename + "_loss.txt");
	for (int i = 0; i < 1000000; i++) {
		Perceptron p = Perceptron(N, lr);
		double loss = 1;
		int epoch;
		if (i%100000 == 0)
			cout << i << "\n";
		for (epoch = 1; loss != 0; epoch++) {
			//cout << "epoch: " << epoch << "\n";
			//p.print_weights();					// weight 출력
			//p.print_linear_function();			// 직선의 방정식 출력
			//p.write_linear_function(lineFile);	// 직선의 방정식 저장
			loss = p.run(x, target);			// perceptron 연산
	//		cout << "loss: " << loss << "\n\n";	// loss 출력
	//		lossFile << loss << "\n";			// loss 저장
			
			// epoch가 10만회를 넘으면 정지
			if (epoch == 100000)
				break;
		}
		epochFile << epoch << "\n";
	}
	epochFile.close();
//	lineFile.close();
//	lossFile.close();
	return 0;
}