#include <ai.h>
#include "mnist/mnist_reader.hpp" // remove later

#define LOG_TO_FILE 1


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//													MNIST Test
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main()
{
#if LOG_TO_FILE
	freopen("../output.txt", "w", stdout);
#endif

	// Load MNIST data
	const bool limitedLoad = false;
	const char* MNIST_DATA_LOCATION = ".//..//..//examples//mnist-cnn//mnist//data";
	mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
		mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION, limitedLoad ? 10 : 0, limitedLoad ? 10 : 0);
	std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
	std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
	std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
	std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

	auto input = make_shared<Input>(Dim { 28, 28, 1 });
	auto x = (*make_shared<Conv>(32, 3, 3, 2, 1))(input);
	x = (*make_shared<Relu>())(x);
	x = (*make_shared<Conv>(64, 3, 3, 2, 1))(x);
	x = (*make_shared<Relu>())(x);
	x = (*make_shared<Dense>(128))(x);
	x = (*make_shared<Dense>(10))(x);
	auto output = make_shared<Softmax>();
	auto loss = make_shared<CrossEntropy>();
	x = (*output)(x);
	x = (*loss)(x);

	Model model({ input }, { loss });

	AdagradTrainer t;
	t.l2Decay = 0.f;
	t.lr = 0.01f;
	t.batchSize = 128;
	t.init(&model);

	const auto imageToTensorFn = [](const std::vector<uint8_t>& image, Tensor& t)
	{
		t.init({ 28,28,1 });
		for (int i = 0; i < image.size(); i++)
		{
			t.data[i] = image[i] / 255.f;
		}
	};

	const auto labelToOneHot = [](uint8_t label, Tensor& t)
	{
		t.init({ 1, 1, 10 });
		t.setZero();
		t.data[label] = 1;
	};

	const int accSamplesCount = 1000;
	const auto calcAcc = [&]()
	{
		int total = 0;
		int correct = 0;
		for (int k = 0; k < accSamplesCount; k++)
		{
			const int index = g_randomGen() % dataset.test_images.size();
			const std::vector<uint8_t>& image = dataset.test_images[index];
			Tensor X;
			imageToTensorFn(image, X);

			model.forward({ X });
			const Tensor& ans = output->Y;
			uint32_t maxIndex = std::distance(ans.data.begin(), std::max_element(ans.data.begin(), ans.data.end()));
			if (dataset.test_labels[index] == maxIndex)
			{
				correct++;
			}
			total++;
		}
		return static_cast<float>(correct) / total;
	};

	const int epochs = 100;
	const int epochSize = 1000;
	int epoch = 0;
	for (int epoch = 0; epoch < epochs; epoch++)
	{
		std::cout << "Epoch:" << epoch << "\n";
		auto start = std::chrono::steady_clock::now();
		for (int k = 0; k < epochSize; k++)
		{
			const int index = g_randomGen() % dataset.training_images.size();
			const std::vector<uint8_t>& image = dataset.training_images[index];
			Tensor X;
			imageToTensorFn(image, X);
			Tensor Y;
			labelToOneHot(dataset.training_labels[index], Y);
			t.train({ X }, { Y });
		}
		auto epochTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
		std::cout << "EpochTime (ms):" << epochTime.count() << "\n";
		std::cout << "Loss:" << t.getLoss() << "\n";
		std::cout << "Acc:" << calcAcc() << "\n";
		std::cout << std::flush;
	}
	return 0;
}
