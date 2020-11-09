// 컴퓨터과학부 2016920029 유시온
// 인공지능 과제 #3 Multi-Layer perceptron 구현
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "activation_functions.h"
#include "objective_functions.h"
#include "Perceptron.h"
#include "Layer.h"
#include "Model.h"

using namespace std;

// main 함수
int main(void) {
	const double TOLERANCE = 0.001;		// tolerance
	const int AND = 1;					// AND 
	const int OR = 2;					// OR
	const int XOR = 3;					// XOR
	const int DONUT = 4;				// DONUT

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

	// 각종 정보 기록할 file 정의
	ofstream lossFile(filename + "_loss.txt");
	ofstream lineFile(filename + "_line.txt");
	ofstream weightFile(filename + "_weight.txt");
	ofstream dotFile(filename + "_dot.txt");

	// 테스트 수행
	cout << "\n====================실행결과====================\n";
	double loss = 1;	// loss값

	// model 학습 시작
	for (int epoch = 1; loss > TOLERANCE; epoch++) {
		cout << "epoch: " << epoch << "\n";
		loss = m.run(x, target, dotFile, false, false);	// 학습 수행

		// 직선 정보 출력
		m.print_linear_function();
		m.write_linear_function(lineFile);

		// loss 출력
		cout << "loss: " << loss << "\n\n";
		lossFile << loss << "\n";
	}

	// 최종 weight 저장
	m.write_weights(weightFile);

	lossFile.close();
	lineFile.close();
	weightFile.close();
	dotFile.close();

	return 0;
}