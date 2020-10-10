// ��ǻ�Ͱ��к� 2016920029 ���ÿ�
// �ΰ����� ���� #3 Multi-Layer perceptron ����
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>

using namespace std;

// Hard Limiting (���ذ�: 0)
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
	// Perceptron ������
	// n: �Է� ����, lr: learning rate
	Perceptron(int n, double lr) {
		this->input_dim = n;				// �Է� ���� �ʱ�ȭ
		this->weight_random_initialize();	// weight �ʱ�ȭ
		this->learning_rate = lr;			// learning_rate �ʱ�ȭ
	}		

	// weight�� ��ȯ�Ѵ�.
	vector<double> get_weights() {
		return weights;
	}
	
	// weight�� �ֿܼ� ����Ѵ�.
	void print_weights() {
		cout << "weights: ";
		for (const double &w : weights) {
			cout << w << " ";
		}
		cout << "\n";
	}

	// weight�� ���� ������ �������� ����Ѵ�.
	// 2���� �Է¿��� ��ȿ
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
	// 2���� �Է¿��� ��ȿ
	void write_linear_function(ofstream &outFile) {
		// �Է� ���� üũ
		if (input_dim != 2)
			return;

		// ������ ������ ����
		outFile << (-1 * weights[0] / weights[1]) << " " << (-1 * weights[2] / weights[1]) << "\n";
	}

	// forward ����
	// x: input���� ������ vector
	// ��ȯ��: forward ���� ���
	double forward(const vector<double> &x) {
		// �ùٸ� �Է����� üũ
		if (x.size() != input_dim) {
			cout << "Perceptron error: x.size() != input_dim.\n";
			exit(-1);
		}

		// forward ���� ����
		double result = 0;

		// �� input�� weight�� ���� ���� ���� ���Ѵ�.
		for (int i = 0; i < input_dim; i++)
			result += x[i] * weights[i];
		result += weights[input_dim];	// Threashold ��

		// net ���� �����صд�. (���� delta �� ���ϴ� �� Ȱ��)
		net = result;

		// activation function
		result = activate(result);
		
		return result;
	}

/*
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
*/
	
	// weight update
	// x: input���� ������ vector, delta_bar: ���� layer���� �Ѿ�� delta_bar
	// ��ȯ��: ���� layer�� �ѱ� delta_bar�� ����ϱ� ���� ���� ��� vector (w * delta)
	vector<double> update_weight(const vector<double> &x, const double &delta_bar) {
		// delta ���
		delta = delta_bar * SigmoidPrime(net);
		
		// ���� layer�� �Ѱ��� ���� ��� (w * delta)
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
*/

private:
	int input_dim = 0;				// input ����
	double learning_rate;			// learning_rate
	vector<double> weights;			// perceptron�� weight ��
	double net;						// net�� (activation �Լ� ���� ��)
	double delta;					// delta��, ���� layer���� �Ѿ�� delta_bar�� ���
/*
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
*/
	
	// weight�� ������ ������ �ʱ�ȭ�Ѵ�.
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

	// Ȱ��ȭ �Լ�
	double activate(double in) {
		return Sigmoid(in);	// Sigmoid �Լ� ����
	//	return hard_limiting(in);
	}
};

// Layer Class
class Layer {
public:
	// ������
	// input_dim: layer�� �Է� ����, output_dim: layer�� ��� ����
	// lr: learning rate
	Layer(int input_dim, int output_dim, double lr) {
		this->input_dim = input_dim;			// �Է� ���� �ʱ�ȭ
		this->output_dim = output_dim;			// ��� ���� �ʱ�ȭ
		
		// layer�� ������ perceptron ����
		for (int i = 0; i < output_dim; i++)
			nodes.push_back(Perceptron(input_dim, lr));
	}
	
	// Layer�� ���ϴ� node���� weight�� file�� ���
	// outFile: weight�� ������ file
	// ��������
	// - Layer ���� �ϳ��� ����� �����Ѵ�.
	// - �� row�� �� ����� weight�� ��Ÿ����. bias�� ���� ������ ����̴�.
	void write_weight(ofstream &outFile) {
		for (Perceptron &n : nodes) {
			for (const double &w : n.get_weights())
				outFile << w << " ";
			outFile << "\n";
		}
	}

	// layer�� ���� nodes���� ������ �������� �ֿܼ� ����Ѵ�.
	void print_linear_function(const int &layer_num) {
		for (int i = 0; i < nodes.size(); i++) {
			cout << layer_num << "-" << i+1 << " ";
			nodes[i].print_linear_function();
		}
	}

	// layer�� ���� nodes���� ������ �������� file�� �����Ѵ�.
	void write_linear_function(ofstream &outFile) {
		for (Perceptron &n: nodes)
			n.write_linear_function(outFile);
		outFile << "\n";
	}

	// ���� forward ���꿡�� ����ߴ� input�� ��ȯ�Ѵ�.
	vector<double> get_prev_x() {
		return prev_x;
	}

	// foward ����
	// x: layer�� ���� �Է�
	// ��ȯ��: forward ���� ���
	vector<double> forward(const vector<double> &x) {
		// �ùٸ� �Է����� üũ
		if (x.size() != input_dim) {
			cout << "Layer error: x.size() != input_dim\n";
			cout << "input.size() == " << x.size() << "\n";
			cout << "input_dim == " << input_dim << "\n";
			exit(-1);
		}

		prev_x = x;			// �Է��� �����صд�. (backward���� ���)
		
		// forward ���� ����
		vector<double> rtn;
		// layer�� ���� perceptron ���� forward ����
		for (Perceptron &p : nodes)
			rtn.push_back(p.forward(x));
		
		return rtn;
	}

	// backward ����
	// delta_bar: ���� layer ���� ��� �� delta_bar vector
	// ��ȯ��: ���� layer�� ���� delta_bar vector
	vector<double> backward(const vector<double> &delta_bar) {
		// �ùٸ� �Է����� üũ
		if (delta_bar.size() != output_dim) {
			cout << "Layer error: delta_bar.size() != output_dim\n";
			exit(-1);
		}

		// backward ���� ����
		vector<double> rtn(input_dim, 0);		// ���� layer�� ���� delta_bar vector
		
		for (int i = 0; i < output_dim; i++) {
			// layer�� ���� perceptron�� ���� delta_bar�� �̿��� weight update
			vector<double> update_delta = nodes[i].update_weight(prev_x, delta_bar[i]);

			// ���� layer�� ���� delta_bar ���
			for (int j = 0; j < input_dim; j++)
				rtn[j] += update_delta[j];
		}

		return rtn;
	}

private:
	vector<Perceptron> nodes;	// layer�� �����ϴ� perceptron vector
	int input_dim;				// �Է� ����
	int output_dim;				// ��� ����
	vector<double> prev_x;		// forward ���꿡 ����� �Է�
};

// Model class
class Model {
public:
	// ������
	// input_dim: �Է� data�� ����, layers_dim: �� layer�� output_dim�� ���� ����
	// lr: learning rate
	Model(int input_dim, const vector<int> &layers_dim, double lr) {
		// �ùٸ� �Է����� üũ
		if (layers_dim.size() == 0) {
			cout << "Model error: layers_dim.size() == 0\n";
			exit(-1);
		}

		// model ����
		layers.push_back(Layer(input_dim, layers_dim[0], lr));
		for (int i = 1; i < layers_dim.size(); i++)
			layers.push_back(Layer(layers_dim[i - 1], layers_dim[i], lr));
	}

	// Model�� �� Layer�� node���� weight�� file�� ����Ѵ�.
	// outFile: weight�� ����� file
	// ��������
	// - Layer ���� �ϳ��� ����� �����Ѵ�.
	// - �� row�� �� ����� weight�� ��Ÿ����. bias�� ���� ������ ����̴�.
	void write_weights(ofstream &outFile) {
		for (int i = 0; i < layers.size(); i++) {
			outFile << "Layer " << i+1 << "\n";
			layers[i].write_weight(outFile);
			outFile << "\n";
		}
	}

	// model�� ���� layer�鿡 ���� �� node�� ������ �������� �ֿܼ� ����Ѵ�.
	// �������: [layer ��ȣ]-[node��ȣ] ������ ������: [������ ������]
	// layer��ȣ�� node ��ȣ ��� 1������ �����Ѵ�.
	void print_linear_function() {
		for (int i = 0; i < layers.size(); i++)
			layers[i].print_linear_function(i);
	}

	// model�� ���� layer�鿡 ���� �� node�� ������ �������� file�� �����Ѵ�.
	// �� epoch�� ","�� �����Ѵ�.
	void write_linear_function(ofstream &outFile) {
		for (Layer &l: layers)
			l.write_linear_function(outFile);
		outFile << ",";
	}

	// �� �Է��� hidden layer�� ���� �̵��� ����Ѵ�.
	// {2, 1}�� �ش��ϴ� �𵨿��� ���������� �۵��Ѵ�.
	// x: model�� ���� �Է�
	void print_dot_moving(const vector<double> &x) {
		// �ùٸ��� ���� ȣ��
		if (layers.size() != 2 || x.size() != 2)
			return;

		vector<double> in = layers[1].get_prev_x();
		if (in.size() != 2) {
			cout << "model error: print_dot_moving�� ������ �Է¿��� �۵��ϴ� �Լ��Դϴ�.\n";
			return;
		}
		cout << "(" << x[0] << ", " << x[1] << ") -> (" << in[0] << ", " << in[1] << ")\n";
	}

	// forward ����
	// x: model�� ���� �Է�
	// ��ȯ��: forward ���� ���
	vector<double> forward(const vector<double> &x) {
		// layer ������� �����Ű�� ����
		vector<double> nx = x;
		for (Layer &l : layers)
			nx = l.forward(nx);

		return nx;
	}

	// backward ����
	// x: �Է� data, y: model�� ������ ���, target: ���� ���
	void backward(const vector<double> &x, const vector<double> &y, const vector<double> &target) {
		// �ʱ� delta_bar �Ի�
		vector<double> delta_bar;
		for (int i = 0; i < y.size(); i++)
			delta_bar.push_back(-target[i] + y[i]);

		// back propagation ����
		for (int i = (int)layers.size() - 1; i >= 0; i--)
			delta_bar = layers[i].backward(delta_bar);		// delta_bar ������Ʈ
	}

	// model �۵�, �� case�� ���ؼ� �н�
	// x: model�� ���� input, target: ���� ���
	// ��ȯ��: model�� ���� ��� (���� ���)
	vector<double> run(const vector<double> &x, const vector<double> &target) {
		// forward ���� ����
		vector<double> y = forward(x);

		// �ùٸ� �Է����� üũ
		if (y.size() != target.size()) {
			cout << "Model error: y.size() != target.size()\n";
			exit(-1);
		}

		// backward ���� ����
		backward(x, y, target);

		return y;
	}

	// model �۵�, �ټ��� case�� ���ؼ� �н�
	// x: model�� ���� input, target: ���� ���
	// print: �ش� epoch���� ����� ���, dot_moving: �� �̵� ��� ��� ({2, 1} �𵨿����� �۵�)
	// ��ȯ��: ������ ����� loss��
	double run(const vector<vector<double>> &input, const vector<vector<double>> &target, const bool &print=true, const bool &dot_moving=true) {
		// �ùٸ� �Է����� üũ
		if (input.size() != target.size()) {
			cout << "Model error: input.size() != target.size()\n";
			exit(-1);
		}

		// model �۵� ����
		double loss = 0;

		// ���̽� ���� ���
		for (int i = 0; i < input.size(); i++) {
			vector<double> predict = run(input[i], target[i]);	// ������ ���
			loss += mean_squared_error(predict, target[i]);		// loss ���

			// �� �̵� ��� ���
			if (dot_moving)
				print_dot_moving(input[i]);

			// ���� ��� ���
			if (print) {
				for (const double &x : input[i])
					cout << x << " ";
				cout << " -> ";
				for (const double &y: predict)
					cout << y << " ";
				cout << "\n";
			}
		}

		return loss;
	}

private:
	vector<Layer> layers;	// model�� �����ϴ� layer vector
};

// main �Լ�
int main(void) {
	const double TOLERANCE = 0.001;
	const int AND = 1;
	const int OR = 2;
	const int XOR = 3;
	const int DONUT = 4;

	// �׽�Ʈ�� data ����
	int select = 0;
	cout << "�׽�Ʈ�� data�� ������\n";
	cout << "1: AND, 2: OR, 3: XOR 4: DONUT\n";
	cout << "�Է�: ";	cin >> select;
	string filename;

	//�׽�Ʈ�� data�� ���� input, target ����
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
		cout << "Data�� �߸� �����߽��ϴ�.\n";
		exit(-1);
	}


	// layer �� ����
	int layer_num = 0;
	cout << "layer ���� ���ϼ���: ";	cin >> layer_num;
	
	// layer ���� ��� ���� ����
	cout << "�� layer�� output ������ ����� �����Ͽ� �Է��ϼ���: ";
	vector<int> layers_dim(layer_num);
	for (int i = 0; i < layer_num; i++)
		cin >> layers_dim[i];

	// �ùٸ� �Է����� üũ
	if (layers_dim[layer_num - 1] != target[0].size()) {
		cout << "���� ��� ������ target�� ��ġ���� �ʽ��ϴ�.\n";
		exit(-1);
	}

	// learning rate �Է�
	double lr;
	cout << "learning rate�� �Է��ϼ���: ";		cin >> lr;

	// model ����
	Model m = Model(x[0].size(), layers_dim, lr);

	// ���� ���� ����� file ����
	ofstream lossFile(filename + "_loss.txt");
	ofstream lineFile(filename + "_line.txt");

	// �׽�Ʈ ����
	cout << "\n====================������====================\n";
	double loss = 1;
	for (int epoch = 1; loss > TOLERANCE; epoch++) {
		cout << "epoch: " << epoch << "\n";
		loss = m.run(x, target);
		if (epoch == 1 || epoch % 1000 == 0) {
			ofstream weightFile(filename + "_weight_epoch_" + to_string(epoch)+".txt");
			m.write_weights(weightFile);
			weightFile.close();
		}
		m.print_linear_function();
		m.write_linear_function(lineFile);

		cout << "loss: " << loss << "\n\n";	// loss ���
		lossFile << loss << "\n";
	}
	
	lossFile.close();
	lineFile.close();
	return 0;
	
	// test ���
	// 1. {4, 1} + lr = 0.5�� ���� �� ��� ������. ������ �� 5õȸ
	// 2. {2, 4, 4, 1} + lr = 0.7�� �ϸ� �� �� ����.
	// TODO
	// 1. ������ �м� (��帶�� ���� �׷���, epoch���� error �׷���)
	// 2. weight�� ��� �������� ���Ͽ� ����(?)
	// 3. ������� �ۼ�
}