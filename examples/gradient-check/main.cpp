#include <ai.h>

#define LOG_TO_FILE 1

static void gradientCheck(NN& net, const Tensor& x, const Tensor& y)
{
	std::vector<OptimizationData> m_optData;
	m_optData.clear();
	net.fillOptimizationData(m_optData);

	const float loss = net.forward(x, y);
#if LOG_TO_FILE
	std::cout << "loss: " << loss << "\n";
#endif
	net.backward(x, y);

	const float epsilon = 1e-3f;

	for (uint32_t i = 0; i < m_optData.size(); i++)
	{
		OptimizationData& optData = m_optData[i];

		for (uint32_t j = 0; j < optData.theta.size(); j++)
		{
			float& theta = optData.theta[j];
			float& dtheta = optData.dtheta[j];

			const float x0 = theta;

			theta = x0 + epsilon;
			const double fplus = net.forward(x, y);
			theta = x0 - epsilon;
			const double fminus = net.forward(x, y);
			theta = x0;

			const double gradApprox = (fplus - fminus) / (2. * epsilon);
			const double relError = std::abs((gradApprox - static_cast<double>(dtheta)) /*/ (gradApprox + static_cast<double>(dtheta))*/);
#if LOG_TO_FILE
			std::cout << i << "," << j << " " << fplus << " " << fminus << " " << relError << "\n";
#endif
			assert(relError < 1e-2);
		}
	}
}

int main()
{
#if LOG_TO_FILE
	freopen("../output.txt", "w", stdout);
#endif

	NN nn;
	nn.addConv(1, 1, 1, 1, 0);
	nn.addConv(1, 1, 1, 1, 0);
	nn.addConv(1, 3, 3, 1, 1);
	nn.addDense(3);
	nn.addDense(3);
	nn.addDense(2);
	nn.addSoftmax();
	nn.setLoss(std::make_shared<CrossEntropy>());

	Tensor input; 
	input.initRand({ 4, 4, 1 });
	nn.initialize(input.dim);

	Tensor ans = nn.forward(input);
	std::cout << ans;

	AdamTrainer t;
	t.l2Decay = 0;
	t.lr = 0.001f;
	t.init(&nn);
	
	Tensor output;
	output.init({ 1, 1, 2 });
	output.data = { 0, 1 };

	gradientCheck(nn, input, output);

	const float epochs = 50;
	for (int i = 0; i < epochs; i++)
	{
		t.train(input, output);
		std::cout << "Loss:" << t.getLoss() << "\n";

	}
	std::cout << nn.forward(input);

	//getchar();

	return 0;
}
