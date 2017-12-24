#include <ai.h>
#define LOG_TO_FILE 1

static void gradientCheck(Model& net, const Tensor& x, const Tensor& y)
{
	std::vector<OptimizationData> m_optData;
	m_optData.clear();
	net.fillOptimizationData(m_optData);

	const float loss = net.forward(x, y);
#if LOG_TO_FILE
	std::cout << "loss: " << loss << "\n";
#endif
	net.backward();

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

	Dim inputDim = { 4,4,1 };
	auto input = make_shared<Input>(inputDim);
	auto x = (*make_shared<Conv>(16, 3, 3, 2, 1))(input);
	x = (*make_shared<Relu>())(x);
	x = (*make_shared<Conv>(16, 3, 3, 2, 1))(x);
	x = (*make_shared<Relu>())(x);
	x = (*make_shared<Dense>(32))(x);
	x = (*make_shared<Dense>(10))(x);
	auto output = make_shared<Softmax>();
	auto loss = make_shared<CrossEntropy>();
	x = (*output)(x);
	x = (*loss)(x);

	Model model({ input }, { loss });

	AdamTrainer t;
	t.l2Decay = 0;
	t.lr = 0.001f;
	t.init(&model);
	
	Tensor X, Y;
	X.initRand(inputDim);
	Y.initZero({ 1, 1, 10 });
	Y.data[2] = 1;

	model.forward(X);
	std::cout << output->Y;

	gradientCheck(model, X, Y);

	const float epochs = 50;
	for (int i = 0; i < epochs; i++)
	{
		t.train(X, Y);
		std::cout << "Loss:" << t.getLoss() << "\n";

	}
	model.forward(X);
	std::cout << output->Y;

	//getchar();

	return 0;
}