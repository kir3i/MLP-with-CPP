// ��ǻ�Ͱ��к� 2016920029 ���ÿ�
// �ΰ����� ���� #2 �Է��� n������ perceptron learning ����
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

// Sigmoid ���Լ�
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
	// ������
	Perceptron(int n, double lr) {
		this->input_dim = n;				// �Է� ���� �ʱ�ȭ
		this->weight_random_initialize();	// weight �ʱ�ȭ
		this->learning_rate = lr;			// learning_rate �ʱ�ȭ
	}		

	// weight�� �ʱ�ȭ�Ѵ�.
	void weight_random_initialize() {
		// random number generator ����
		random_device rd;
		mt19937 e2(rd());
		uniform_real_distribution<> dist(-1, 1);

		// random number�� weight �ʱ�ȭ
		weights.resize(input_dim + 1);		// �ܺο� ������ �ʴ� weight �ϳ� �߰� (threshold)
		for (int i = 0; i < input_dim + 1; i++)
			weights[i] = dist(e2);
	}

	// weight�� ����Ѵ�.
	void print_weights() {
		cout << "weights: ";
		for (const double &w : weights) {
			cout << w << " ";
		}
		cout << "\n";
	}

	// weight�� ���� ������ �������� ����Ѵ�.
	// (2���� �Է¿��� ��ȿ��)
	void print_linear_function() {
		// �Է� ���� üũ
		if (input_dim != 2)
			return;

		// ������ ������ ���
		cout << "������ ������: x2 = ";
		cout << (weights[0] / weights[1] > 0 ? "-" : "") << abs(weights[0] / weights[1]) << " * x1";
		cout << (weights[2] / weights[1] > 0 ? " - " : " + ") << abs(weights[2] / weights[1]) << "\n";
	}

	// ������ �������� ���Ϸ� �����Ѵ�.
	void write_linear_function(ofstream &outFile) {
		// �Է� ���� üũ
		if (input_dim != 2)
			return;

		// ������ ������ ����
		outFile << (-1 * weights[0] / weights[1]) << " " << (-1 * weights[2] / weights[1]) << "\n";
	}

	// forward ����
	// x: input���� ������ vector
	// ��ȯ��: ���� ���
	double forward(const vector<double> &x) {
		// �ùٸ� �Է����� üũ
		if (x.size() != input_dim) {
			cout << "Perceptron�� input_dim�� input_vals.size()�� ��ġ���� �ʽ��ϴ�.\n";
			exit(-1);
		}

		// forward ���� ����
		double result = 0;
		// �� input�� weight�� ���� ���� ���� ���Ѵ�.
		for (int i = 0; i < input_dim; i++)
			result += x[i] * weights[i];
		result += weights[input_dim];	// Threashold ��

		net = result;

		// activation function
		result = activate(result);
		
		return result;
	}

	// weight update(learning), �� ���� case�� ���� ������Ʈ�Ѵ�.
	// back propagation�� ���� weight�� �����Ѵ�.
	// x: input���� ������ vector, y: �����, target: �ùٸ� �����
	void update_weight(const vector<double> &x, const double &y, const double &target) {
		for (int i = 0; i < input_dim; i++)
			weights[i] += learning_rate * (target - y) * x[i];
		weights[input_dim] += learning_rate * (target - y);
	}

	// weight update(learning), ���� ���� case�� ���� ������Ʈ�Ѵ�.
	// back propagation�� ���� weight�� �����Ѵ�.
	// x: input���� ������ vector, y: �����, target: �ùٸ� �����
	void update_weight(const vector<vector<double>> &input, const vector<double> &y, const vector<double> &target) {
		for (int i = 0; i < input.size(); i++)
			update_weight(input[i], y[i], target[i]);
	}

	vector<double> update_weight(const vector<double> &x, const double &delta_bar) {
		delta = delta_bar * SigmoidPrime(net);
		
		// ���� ���̾ �Ѱ��� ���� ���
		vector<double> rtn;
		for (int i = 0; i < input_dim; i++)
			rtn.push_back(weights[i] * delta);

		// weight ������Ʈ
		for (int i = 0; i < input_dim; i++)
			weights[i] += (-learning_rate * delta * x[i]);
		
		// threshold ������Ʈ
		weights[input_dim] += (-learning_rate * delta);

		return rtn;
	}

	// Perceptron �۵� (����)
	// input: input���� ������ vector, target: �ùٸ� �����
	// ��ȯ��: Perceptron ���� ����� ���� loss ��
	double run(const vector<vector<double>> &input, const vector<double> &target) {
		// �ùٸ� �Է����� üũ
		if (input.size() != target.size()) {
			cout << "input�� target�� ������ ��ġ���� �ʽ��ϴ�.\n";
			exit(-1);
		}

		// Perceptron ���� ����
		double loss = 0;
		vector<double> y(input.size());
		// case���� ����
		for (int i = 0; i < input.size(); i++) {
			y[i] = forward(input[i]);					// forward ����
			loss += (target[i] - y[i]) * (target[i] - y[i]) / 2;	// mean squared error
			//update_weight(input[i], y[i], target[i]);
		}
		update_weight(input, y, target);							// backward ����
		return loss;
	}

private:
	int input_dim = 0;				// input ����
	double learning_rate;			// learning_rate
	vector<double> weights;			// perceptron�� weight ��
	double net;
	double delta;

	// weight�� ���ϴ� ������ �����Ѵ�.
	// w: �����ϰ��� �ϴ� weight ���� ���� vector
	void set_weights(const vector<double> &w) {

		// �ùٸ� �Է����� üũ
		if (w.size() != input_dim) {
			cout << "Perceptron�� input_dim�� w.size()�� ��ġ���� �ʽ��ϴ�.\n";
			return;
		}

		// �Է°��� weights�� ����
		for (int i = 0; i < input_dim; i++)
			weights[i] = w[i];
	}

	// Ȱ��ȭ �Լ�
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

	// delta_bar: ���� ���̾� ���� ��� �� delta_bar ��
	vector<double> backward(const vector<double> &x, const vector<double> &delta_bar) {
		if (delta_bar.size() != output_dim) {
			cout << "Layer error: delta_bar.size() != output_dim\n";
			exit(-1);
		}

		// ���� ���̾ ���� delta_bar
		vector<double> rtn(input_dim, 0);

		for (int i = 0; i < output_dim; i++) {
			vector<double> update_delta = nodes[i].update_weight(x, delta_bar[i]);

			// ���� ���̾ ���� delta_bar ������Ʈ
			for (int j=0; j<input_dim; j++)
				rtn[j] += update_delta[j];
		}

		return rtn;
	}

	double run(const vector<vector<double>> &input, const vector<double> &target) {
		// �ùٸ� �Է����� üũ
		if (input.size() != target.size()) {
			cout << "input�� target�� ������ ��ġ���� �ʽ��ϴ�.\n";
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
	// input_dim: �Է� data�� ����
	// layers_dim: �� layer�� output_dim�� ���� ����
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
		// ������ delta_bar
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

// main �Լ�
int main(void) {
	const int AND = 1;
	const int OR = 2;
	const int XOR = 3;
	const int DONUT = 4;

	// �Է� ������ 2�� �����Ѵ�.
//	int N = 2;	//cin >> N;
//	if (N <= 0)	return 0;

	// learning rate �Է�
	double lr;	
	cout << "learning rate�� �Է��ϼ���: ";		cin >> lr;

	// �׽�Ʈ�� gate ����
	int select = 0;
	cout << "�׽�Ʈ�� gate�� ������\n";
	cout << "1: AND, 2: OR, 3: XOR\n";
	cout << "�Է�: ";	cin >> select;
	string filename;




	//�׽�Ʈ�� gate�� ���� input, target ����
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
		cout << "Gate�� �߸� �����߽��ϴ�.\n";
		exit(-1);
	}


	vector<int> layers_dim = { 2, 1 };
	Model m = Model(x[0].size(), layers_dim, lr);

	// �׽�Ʈ ����
	cout << "\n====================������====================\n";
	double loss = 1;
	for (int epoch = 1; loss > 0.01; epoch++) {
		cout << "epoch: " << epoch << "\n";
		loss = m.run(x, target);
		cout << "loss: " << loss << "\n\n";	// loss ���
	}
	

/*

	// Perceptron ����
	Perceptron p = Perceptron(x[0].size(), lr);
//	ofstream lineFile(filename + ".txt");
//	ofstream lossFile(filename + "_loss.txt");
	double loss = 1;
	for (int epoch = 1; loss > 0.1; epoch++) {
		cout << "epoch: " << epoch << "\n";
		p.print_weights();					// weight ���
		p.print_linear_function();			// ������ ������ ���
		//p.write_linear_function(lineFile);	// ������ ������ ����
		loss = p.run(x, target);			// perceptron ����
		cout << "loss: " << loss << "\n\n";	// loss ���
		//lossFile << loss << "\n";			// loss ����
			
		// epoch�� 10��ȸ�� ������ ����
		if (epoch >= 100000)
			break;
	}
//	lineFile.close();
//	lossFile.close();
	return 0;
*/
}