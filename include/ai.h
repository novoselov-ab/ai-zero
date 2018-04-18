#pragma once

#undef NDEBUG
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <set>
#include <random>
#include <iostream>
#include <functional>
#include <cassert>
#include <chrono>
#include <future>
#include <iomanip>
#include <ctime>
#include <ostream>
#include <sstream>
#include <istream>
#include <fstream>
#include <filesystem>


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//														Common
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;
using namespace std::chrono;
namespace fs = std::experimental::filesystem;
std::default_random_engine g_randomGen;

// Serialization utils
// ====================

template<class T>
void serializePODVector(std::ostream& os, const vector<T>& v)
{
	os.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
}

template<class T>
void deserializePODVector(vector<T>& v, std::istream& is)
{
	is.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(T));
}

template<class T>
void serializePOD(std::ostream& os, const T& v)
{
	os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template<class T>
void deserializePOD(T& v, std::istream& is)
{
	is.read(reinterpret_cast<char*>(&v), sizeof(T));
}


// Dim (dimensions triple)
// ====================

struct Dim
{
	uint32_t sx;
	uint32_t sy;
	uint32_t depth;

	uint32_t size() const
	{
		return sx * sy * depth;
	}

	bool operator==(const Dim& other) const
	{
		return sx == other.sx && sy == other.sy && depth == other.depth;
	}

	void serialize(std::ostream& os) const
	{
		serializePOD(os, *this);
	}

	void deserialize(std::istream& is)
	{
		deserializePOD(*this, is);
	}
};


// Tensor (3D array)
// ====================

struct Tensor
{
	Tensor(Dim d = Dim()) : dim(d)
	{
		data.resize(dim.size());
	}

	void init(Dim d)
	{
		dim = d;
		data.resize(dim.size());
	}

	void initZero(Dim d)
	{
		init(d);
		setZero();
	}

	void initOnes(Dim d)
	{
		init(d);
		setOnes();
	}

	void initRand(Dim d)
	{
		init(d);
		setRand();
	}

	void setZero()
	{
		std::fill(data.begin(), data.end(), 0.f);
	}

	void setOnes()
	{
		std::fill(data.begin(), data.end(), 1.f);
	}

	void setRand()
	{
		const float scale = std::sqrt(1.0f / dim.size());
		std::normal_distribution<float> distribution(0.f, scale);
		for (auto& x : data)
		{
			x = distribution(g_randomGen);
		}
	}

	float get(uint32_t x, uint32_t y, uint32_t d) const
	{
		return data[(dim.sx * y + x) * dim.depth + d];
	}

	void set(uint32_t x, uint32_t y, uint32_t d, float v)
	{
		data[(dim.sx * y + x) * dim.depth + d] = v;
	}

	void add(uint32_t x, uint32_t y, uint32_t d, float v)
	{
		set(x, y, d, (get(x, y, d) + v));
	}

	friend std::ostream& operator<<(std::ostream& os, const Tensor& t)
	{
		os << "Dim: [" << t.dim.sx << "," << t.dim.sy << "," << t.dim.depth << "]\n";
		for (uint32_t d = 0; d < t.dim.depth; d++)
		{
			if (t.dim.depth > 1)
				os << "depth:" << d << "\n";
			for (uint32_t y = 0; y < t.dim.sy; y++)
			{
				for (uint32_t x = 0; x < t.dim.sx; x++)
				{
					os << /*std::setw(2) << */t.get(x, y, d) << " ";
				}
				os << "\n";
			}
		}
		return os;
	}

	float& operator[](size_t idx) 
	{ 
		return data[idx]; 
	}
	
	const float& operator[](size_t idx) const 
	{ 
		return data[idx]; 
	}

	Tensor& operator+=(const Tensor& t)
	{
		assert(t.dim == dim);
		for (uint32_t i = 0; i < data.size(); i++)
			data[i] += t.data[i];
		return *this;
	}

	void serialize(std::ostream& os) const
	{
		serializePOD(os, dim);
		serializePODVector(os, data);
	}

	void deserialize(std::istream& is)
	{
		deserializePOD(dim, is);
		init(dim);
		deserializePODVector(data, is);
	}

	Dim dim;
	vector<float> data;
};


// Timer
// ====================

struct SimpleTimer
{
	SimpleTimer()
	{
		m_start = steady_clock::now();
	}

	int64_t getElapsedMS() const
	{
		return duration_cast<milliseconds>(steady_clock::now() - m_start).count();
	}

	float getElapsedSeconds() const
	{
		return getElapsedMS() / 1000.0f;
	}

private:
	time_point<steady_clock> m_start;
};


// Random Utils
// ====================

bool randBernoulli(double p = 0.5)
{
	std::bernoulli_distribution d(p);
	return d(g_randomGen);
}

float randUniform(float a, float b)
{
	std::uniform_real_distribution<float> d(a, b);
	return d(g_randomGen);
}

int randUniform(int a, int b)
{
	std::uniform_int_distribution<int> d(a, b);
	return d(g_randomGen);
}

uint32_t randChoice(const Tensor& probs)
{
	float x = randUniform(0.0f, 1.0f);
	float s = 0.f;
	for (uint32_t i = 0; i < probs.data.size(); i++)
	{
		s += probs.data[i];
		if (x < s)
		{
			return i;
		}
	}
	assert(false);
	return 0;
}

void fillRandDirichlet(Tensor& y, float alpha)
{
	// take samples from Gamma distribution (alpha, 1.0)
	std::gamma_distribution<float> d(alpha, 1.0f);
	for (uint32_t i = 0; i < y.data.size(); i++)
		y.data[i] = d(g_randomGen);
	// then norm by sum
	float ys = accumulate(y.data.begin(), y.data.end(), 0.f);
	for (uint32_t i = 0; i < y.data.size(); i++)
		y.data[i] /= ys;
}


// Helpers
// ====================

uint32_t argmax(const Tensor& t)
{
	return static_cast<uint32_t>(distance(t.data.begin(), max_element(t.data.begin(), t.data.end())));
}

std::string dateTimeNow()
{
	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);

	std::ostringstream oss;
	oss << std::put_time(&tm, "%Y-%m-%d %H-%M-%S");
	return oss.str();
}

std::ostream& operator<<(std::ostream& os, const vector<float>& v)
{
	os << "[";
	for (auto x : v)
	{
		os << x << ",";
	}
	os << "]\n";
	return os;
}

void checkCreateDir(fs::path dirPath)
{
	if (!fs::exists(dirPath))
		fs::create_directories(dirPath);
}

static set<string, greater<string>> getDateDescendingSortedFiles(const fs::path& p)
{
	set<string, greater<string>> s;
	for (auto& p : fs::directory_iterator(p))
		s.insert(p.path().string());
	return s;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//														Layers
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct OptimizationData
{
	vector<float>& theta;
	vector<float>& dtheta;
	float l1;
	float l2;
};

class Layer : public std::enable_shared_from_this<Layer>
{
public:
	Tensor Y;
	Tensor dX;

	vector<shared_ptr<Layer>> inputs;
	vector<weak_ptr<Layer>> outputs;

	shared_ptr<Layer> operator()(shared_ptr<Layer> inputLayer)
	{
		return link(inputLayer);
	}

	shared_ptr<Layer> operator()(initializer_list<shared_ptr<Layer>> inputLayers)
	{
		return link(inputLayers);
	}

	shared_ptr<Layer> link(shared_ptr<Layer> inputLayer)
	{
		linkImpl(inputLayer);
		return shared_from_this();
	}

	shared_ptr<Layer> link(initializer_list<shared_ptr<Layer>> inputLayers)
	{
		for (auto l : inputLayers)
			linkImpl(l);
		return shared_from_this();
	}

	virtual void forward() = 0;
	virtual void backward() = 0;

	virtual void fillOptimizationData(vector<OptimizationData>& data) {};

	virtual ~Layer() {}

protected:
	virtual void initialize(Dim inDim) = 0;

private:

	void linkImpl(shared_ptr<Layer> inputLayer)
	{
		assert(inputLayer->Y.dim.size() > 0);
		assert(!inputs.empty() ? (inputs.back()->Y.dim == inputLayer->Y.dim) : true);
		inputs.push_back(inputLayer);
		inputLayer->outputs.push_back(shared_from_this());

		initialize(inputLayer->Y.dim);
	}
};

class SingleInputLayer : public Layer
{
public:
	virtual void forward() override
	{
		assert(inputs.size() <= 1);
		if (inputs.size() > 0)
		{
			forward(inputs[0]->Y);
		}
		else
		{
			Tensor empty;
			forward(empty);
		}
	}

	virtual void backward() override
	{
		Tensor empty;
		if (inputs.size() > 0)
		{
			const Tensor* X = inputs.size() > 0 ? &inputs[0].get()->Y : &empty;
			if (outputs.size() > 0)
			{
				const Tensor* dY = nullptr;
				if (outputs.size() > 1)
				{
					m_dY.initZero(outputs[0].lock().get()->dX.dim);
					m_dY.setZero();
					for (auto o : outputs)
					{
						const auto& dX = o.lock().get()->dX;
						m_dY += dX;
					}
					dY = &m_dY;
				}
				else
				{
					dY = &outputs[0].lock()->dX;
				}
				backward(*X, *dY);
			}
			else
			{
				backward(*X, empty);
			}
		}
	}

	virtual void forward(const Tensor& X) = 0;
	virtual void backward(const Tensor& X, const Tensor& dY) = 0;

private:
	Tensor m_dY;
};

class Input : public SingleInputLayer
{
public:
	Input(Dim inDim)
	{
		Y.initZero(inDim);
		dX.initZero(inDim);
	}

	void initialize(Dim inDim) override {}

	void forward(const Tensor& X) override
	{
	}

	void backward(const Tensor& X, const Tensor& dY) override
	{
		dX = dY;
	}
};

class DotLayer : public SingleInputLayer
{
public:
	float l1 = 0.f;
	float l2 = 1.f;

	void fillOptimizationData(vector<OptimizationData>& data) override
	{
		for (uint32_t i = 0; i < m_filters.size(); i++)
		{
			data.push_back(OptimizationData{ m_filters[i].data, m_filtersGrad[i].data, l1, l2 });
		}
		data.push_back(OptimizationData{ m_biases.data, m_biasesGrad.data, 0.f, 0.f });
	}

protected:
	vector<Tensor> m_filters;
	vector<Tensor> m_filtersGrad;
	Tensor m_biases;
	Tensor m_biasesGrad;
};

class Dense : public DotLayer
{
public:
	Dense(uint32_t n)
	{
		Y.initZero({ 1, 1, n });
		m_filters.resize(n);
		m_filtersGrad.resize(n);
		m_biases.initRand({ 1, 1, n });
		m_biasesGrad.initZero({ 1, 1, n });
	}

	void initialize(Dim inDim) override
	{
		dX.initZero(inDim);
		for (auto& f : m_filters)
			f.initRand({ 1, 1, inDim.size() });
		for (auto& f : m_filtersGrad)
			f.initZero({ 1, 1, inDim.size() });
	}

	void forward(const Tensor& X) override
	{
		for (uint32_t i = 0; i < Y.dim.depth; i++)
		{
			float a = 0.f;
			const auto& f = m_filters[i];
			for (uint32_t k = 0; k < f.dim.depth; k++)
			{
				a += X[k] * f[k];
			}
			a += m_biases[i];
			Y[i] = a;
		}
	}

	void backward(const Tensor& X, const Tensor& dY) override
	{
		dX.setZero();
		for (uint32_t i = 0; i < Y.dim.depth; i++)
		{
			const auto& f = m_filters[i];
			auto& fg = m_filtersGrad[i];
			for (uint32_t k = 0; k < f.dim.depth; k++)
			{
				dX[k] += f[k] * dY[i];
				fg[k] += X[k] * dY[i];
			}
			m_biasesGrad[i] += dY[i];
		}
	}
};

class Conv : public DotLayer
{
public:
	Conv(uint32_t numFilters, uint32_t kx, uint32_t ky, uint32_t stride, uint32_t pad)
	{
		m_numFilters = numFilters;
		m_kx = kx;
		m_ky = ky;
		m_stride = stride;
		m_pad = pad;
	}

	void initialize(Dim inDim) override
	{
		Dim outDim;
		outDim.depth = m_numFilters;
		outDim.sx = (uint32_t)std::floor(((float)inDim.sx + m_pad * 2 - m_kx) / m_stride + 1);
		outDim.sy = (uint32_t)std::floor(((float)inDim.sy + m_pad * 2 - m_ky) / m_stride + 1);

		Y.initZero(outDim);
		dX.initZero(inDim);

		m_filters.resize(outDim.depth);
		m_filtersGrad.resize(outDim.depth);
		m_biases.initRand({ 1, 1, outDim.depth });
		m_biasesGrad.initZero({ 1, 1, outDim.depth });

		for (auto& f : m_filters)
			f.initRand({ m_kx, m_ky, inDim.depth });
		for (auto& f : m_filtersGrad)
			f.initZero({ m_kx, m_ky, inDim.depth });
	}

	void forward(const Tensor& X) override
	{
		const int in_sx = X.dim.sx;
		const int in_sy = X.dim.sy;
		for (uint32_t d = 0; d < Y.dim.depth; d++)
		{
			const Tensor& f = m_filters[d];
			int y = -m_pad;
			for (uint32_t ay = 0; ay < Y.dim.sy; ay++)
			{
				int x = -m_pad;
				for (uint32_t ax = 0; ax < Y.dim.sx; ax++)
				{
					float a = .0f;

					for (uint32_t fy = 0; fy < f.dim.sy; fy++)
					{
						int oy = y + fy;
						for (uint32_t fx = 0; fx < f.dim.sx; fx++)
						{
							int ox = x + fx;
							if (oy >= 0 && oy < in_sy && ox >= 0 && ox < in_sx)
							{
								for (uint32_t fd = 0; fd < f.dim.depth; fd++)
								{
									a += f.get(fx, fy, fd) * X.get(ox, oy, fd);
								}
							}

						}
					}

					a += m_biases[d];
					Y.set(ax, ay, d, a);

					x += m_stride;
				}
				y += m_stride;
			}
		}
	}

	void backward(const Tensor& X, const Tensor& dY) override
	{
		dX.setZero();

		const int in_sx = X.dim.sx;
		const int in_sy = X.dim.sy;
		for (uint32_t d = 0; d < Y.dim.depth; d++)
		{
			const Tensor& f = m_filters[d];
			Tensor& fg = m_filtersGrad[d];
			int y = -m_pad;
			for (uint32_t ay = 0; ay < Y.dim.sy; ay++)
			{
				int x = -m_pad;
				for (uint32_t ax = 0; ax < Y.dim.sx; ax++)
				{
					float chain_grad = dY.get(ax, ay, d);
					for (uint32_t fy = 0; fy < fg.dim.sy; fy++)
					{
						int oy = y + fy;
						for (uint32_t fx = 0; fx < fg.dim.sx; fx++)
						{
							int ox = x + fx;
							if (oy >= 0 && oy < in_sy && ox >= 0 && ox < in_sx)
							{
								for (uint32_t fd = 0; fd < fg.dim.depth; fd++)
								{
									fg.add(fx, fy, fd, (X.get(ox, oy, fd) * chain_grad));
									dX.add(ox, oy, fd, (f.get(fx, fy, fd) * chain_grad));
								}
							}
						}
					}
					m_biasesGrad[d] += chain_grad;

					x += m_stride;
				}
				y += m_stride;
			}
		}
	}

private:
	uint32_t m_numFilters;
	uint32_t m_kx;
	uint32_t m_ky;
	uint32_t m_stride;
	int m_pad;
};

class Relu : public SingleInputLayer
{
public:
	void initialize(Dim inDim) override
	{
		Y.initZero(inDim);
		dX.initZero(inDim);
	}

	void forward(const Tensor& X) override
	{
		for (uint32_t i = 0; i < Y.dim.size(); i++)
		{
			Y[i] = X[i] < 0 ? 0 : X[i];
		}
	}

	void backward(const Tensor& X, const Tensor& dY) override
	{
		for (uint32_t i = 0; i < Y.dim.size(); i++)
		{
			dX[i] = Y[i] <= 0 ? 0 : dY[i];
		}
	}
};

class Softmax : public SingleInputLayer
{
public:
	void initialize(Dim inDim) override
	{
		Y.initZero(inDim);
		dX.initZero(inDim);
	}

	void forward(const Tensor& X) override
	{
		// max act
		float amax = *std::max_element(X.data.begin(), X.data.end());

		// compute exp
		float esum = 0.f;
		for (uint32_t i = 0; i < Y.dim.depth; i++)
		{
			const float e = std::exp(X[i] - amax);
			esum += e;
			Y[i] = e;
		}

		// norm
		for (uint32_t i = 0; i < Y.dim.depth; i++)
		{
			Y[i] /= esum;
		}
	}

	void backward(const Tensor& X, const Tensor& dY) override
	{
		for (uint32_t i = 0; i < Y.dim.depth; i++)
		{
			dX[i] = 0.f;
			for (uint32_t k = 0; k < Y.dim.depth; k++)
			{
				const float df = (k == i) ? Y[i] * (1.f - Y[i]) : -Y[k] * Y[i];
				dX[i] += dY[k] * df;
			}
		}
	}
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//														Loss
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class LossLayer : public SingleInputLayer
{
public:
	Tensor T; // Target (Label)

	void initialize(Dim inDim) override
	{
		Y.initZero({1,1,1});
		dX.initZero(inDim);
		T.initZero(inDim);
	}
};

struct MSE : public LossLayer
{
	void forward(const Tensor& X) override
	{
		float loss = 0.f;
		const uint32_t n = X.dim.size();
		for (uint32_t i = 0; i < n; i++)
		{
			float dy = (X[i] - T[i]);
			loss += dy * dy;
		}
		loss /= n;
	
		Y[0] = loss;
	}

	void backward(const Tensor& X, const Tensor& dY) override
	{
		const uint32_t n = X.dim.size();
		const float factor = 2.f / n;
		for (uint32_t i = 0; i < n; i++)
		{
			float dy = (X[i] - T[i]);
			dX[i] = factor * dy;
		}
	}
};

struct CrossEntropy : public LossLayer
{
	void forward(const Tensor& X) override
	{
		float loss = 0.f;
		const uint32_t n = X.dim.size();
		for (uint32_t i = 0; i < n; i++)
		{
			loss += -T[i] * std::log(X[i]) - (1.f - T[i]) * std::log((1.f - X[i]));
		}

		if (std::isnan(loss))
			loss = 0.f;

		Y[0] = loss;
	}

	void backward(const Tensor& X, const Tensor& dY) override
	{
		const uint32_t n = X.dim.size();
		for (uint32_t i = 0; i < n; i++)
		{
			dX[i] = (X[i] - T[i]) / (X[i] * (1.f - X[i]));
			if (std::isnan(dX[i]))
				dX[i] = 0.f;
			if (std::isinf(dX[i]))
				dX[i] = 1.f;
		}
	}
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//														Model
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Model
{
public:
	Model(vector<shared_ptr<Input>> inputs, vector<shared_ptr<LossLayer>> losses)
		: m_inputs(inputs)
		, m_losses(losses)
	{
		assert(!m_inputs.empty());
		buildPath();
	}

	void forward(const Tensor& X)
	{
		forward({ &X });
	}

	float forward(const Tensor& X, const Tensor& Y)
	{
		return forward({ &X }, { &Y });
	}

	float forward(const vector<const Tensor*>& X, const vector<const Tensor*>& Y = vector<const Tensor*>())
	{
		// set inputs
		assert(m_inputs.size() == X.size());
		for (uint32_t i = 0; i < m_inputs.size(); i++)
		{
			m_inputs[i]->Y = *X[i];
		}

		// set targets
		assert(m_losses.size() >= Y.size());
		for (uint32_t i = 0; i < Y.size(); i++)
		{
			m_losses[i]->T = *Y[i];
		}

		// forward path
		for (auto l : m_forwardPath)
		{
			l->forward();
		}

		// calc loss
		float totalLoss = 0.f;
		for (auto l : m_losses)
		{
			totalLoss += l->Y.data[0];
		}
		return totalLoss;
	}

	void backward()
	{
		for (auto it = m_forwardPath.rbegin(); it != m_forwardPath.rend(); ++it) 
		{
			(*it)->backward();
		}
	}

	void fillOptimizationData(vector<OptimizationData>& data)
	{
		for (auto l : m_forwardPath)
		{
			l->fillOptimizationData(data);
		}
	}

	void save(ostream& os)
	{
		vector<OptimizationData> data;
		fillOptimizationData(data);
		for (auto& d : data)
		{
			serializePODVector(os, d.theta);
		}
	}

	void save(const fs::path& p)
	{
		ofstream ofs(p, std::ifstream::out | std::ios::binary);
		save(ofs);
	}

	void load(istream& is)
	{
		vector<OptimizationData> data;
		fillOptimizationData(data);
		for (auto& d : data)
		{
			deserializePODVector(d.theta, is);
		}
	}

	void load(const fs::path& p)
	{
		std::ifstream ifs(p, std::ifstream::in | std::ios::binary);
		load(ifs);
	}

private:
	void buildPath()
	{
		// build forward path by taking dependencies into account (topological sort)
		m_forwardPath.clear();
		map<Layer*, int> deps;

		for (auto l : m_inputs)
		{
			m_forwardPath.push_back(l.get());
		}

		for (uint32_t i = 0; i < m_forwardPath.size(); i++)
		{
			auto layer = m_forwardPath[i];
			for (auto wp : layer->outputs)
			{
				auto next = wp.lock().get();
				const size_t n = next->inputs.size();
				if (n > 1)
					deps[layer]++;

				if (n == 1 || n == deps[next])
					m_forwardPath.push_back(next);
			}
		}

	}

	vector<Layer*>					m_forwardPath;

	vector<shared_ptr<Input>>		m_inputs;
	vector<shared_ptr<LossLayer>>	m_losses;
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//													Optimizers
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class OptimizerT, class OptimizerSettingsT>
class Trainer
{
public:

	float lr = 0.01f;
	uint32_t batchSize = 1;
	float l1Decay = 0.f;
	float l2Decay = 0.01f;
	OptimizerSettingsT optimizerSettings;

	uint32_t m_iter;

	void init(Model* net)
	{
		m_net = net;
		reset();
	}

	void reset()
	{
		m_iter = 0;
		m_optData.clear();
		m_net->fillOptimizationData(m_optData);
		m_kernels.resize(m_optData.size());
		for (uint32_t i = 0; i < m_optData.size(); i++)
		{
			m_kernels[i].resize(m_optData[i].theta.size());
		}
	}

	void train(const Tensor& X, const Tensor& Y)
	{
		train({ &X }, { &Y });
	}

	void train(const vector<const Tensor*>& X, const vector<const Tensor*>& Y)
	{
		float cost_loss = m_net->forward(X, Y);
		m_net->backward();
		m_lossAcc += cost_loss;

		m_iter++;
		if (m_iter % batchSize == 0)
		{
			vector<std::future<float>> futures;
			for (uint32_t i = 0; i < m_optData.size(); i++)
			{
				futures.push_back(std::async([&](uint32_t i)
				{
					float loss = 0.f;
					OptimizationData& optData = m_optData[i];
#if 0
					std::cout << "" << i << ": theta: " << optData.theta;
					std::cout << "" << i << ": dtheta: " << optData.dtheta;
#endif

					const float l1 = optData.l1 * l1Decay;
					const float l2 = optData.l2 * l2Decay;

					for (uint32_t j = 0; j < optData.theta.size(); j++)
					{
						float& theta = optData.theta[j];
						float& dtheta = optData.dtheta[j];

						loss += l1 * std::abs(theta);
						loss += l2 * theta * theta * 0.5f;

						const float l1grad = l1 * (theta > 0 ? 1.f : -1.f);
						const float l2grad = l2 * (theta);

						const float gtotal = (l1grad + l2grad + dtheta) / batchSize;

						m_kernels[i][j].step(theta, this, gtotal);
						dtheta = 0.f; // zero out gradients
					}

					return loss;
				}, i));
			}

			m_loss = 0.f;
			for (auto &e : futures)
			{
				m_loss += e.get();
			}
			m_loss += m_lossAcc;
			m_lossAcc = 0.f;
		}
	}

	float getLoss() const
	{
		return m_loss;
	}

private:
	Model* m_net;
	vector<OptimizationData> m_optData;
	vector<vector<OptimizerT>> m_kernels;
	float m_loss;
	float m_lossAcc;
};

struct Adagrad
{
	struct Settings
	{
		float eps = 1e-8f;
	};

	Adagrad() : gsum(0.f) {}

	void step(float& theta, const Trainer<Adagrad, Adagrad::Settings>* t, float dtheta)
	{
		gsum = gsum + dtheta * dtheta;
		const float dx = -t->lr / std::sqrt(gsum + t->optimizerSettings.eps) * dtheta;
		theta += dx;
	}

private:
	float gsum;
};
using AdagradTrainer = Trainer<Adagrad, Adagrad::Settings>;

struct Adam
{
	struct Settings
	{
		float eps = 1e-8f;
		float beta1 = 0.9f;
		float beta2 = 0.999f;
	};

	Adam() : m(0.f), v(0.f) {}

	void step(float& theta, const Trainer<Adam, Adam::Settings>* t, float dtheta)
	{
		const Settings& s = t->optimizerSettings;
		m = m * s.beta1 + (1.f - s.beta1) * dtheta;
		v = v * s.beta2 + (1.f - s.beta2) * dtheta * dtheta;
		const float biasCorr1 = m * (1.f - std::powf(s.beta1, static_cast<float>(t->m_iter)));
		const float biasCorr2 = v * (1.f - std::pow<float>(s.beta2, static_cast<float>(t->m_iter)));
		const float dx = -t->lr * biasCorr1 / (std::sqrt(biasCorr2) + s.eps);
		theta += dx;
	}
private:
	float m;
	float v;
};
using AdamTrainer = Trainer<Adam, Adam::Settings>;

struct SGD
{
	struct Settings
	{
	};

	void step(float& theta, const Trainer<SGD, SGD::Settings>* t, float dtheta)
	{
		theta += -t->lr * dtheta;
	}
};
using SGDTrainer = Trainer<SGD, SGD::Settings>;


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//												 General RL Environment
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Game
{
public:
	virtual ~Game() {};
	virtual uint32_t getPlayerCount() const = 0;
	virtual uint32_t getActionCount() const = 0;
	virtual Dim getStateDim() const = 0;
	virtual uint32_t getCurrentPlayer() const = 0;
	virtual uint32_t getTurn() const = 0;
	virtual bool isFinished() const = 0;
	virtual float getReward(uint32_t player) const = 0;
	virtual const Tensor& getState(uint32_t player) const = 0;
	virtual void doAction(const Tensor& action) = 0;
	virtual void getLegalActions(Tensor& legalActions) const = 0;
	virtual void reset() = 0;
	virtual unique_ptr<Game> clone() const = 0;
	virtual void copyFrom(const Game& other) = 0;
	virtual void render() {}

	void doAction(uint32_t actionIndex)
	{
		Tensor action;
		action.initZero({ 1, 1, getActionCount() });
		action[actionIndex] = 1.f;
		doAction(action);
	}
};

class Player
{
public:
	virtual ~Player() {}
	virtual void beginGame() = 0;
	virtual void notifyGameAction(uint32_t action) = 0;
	virtual uint32_t chooseAction(const Game& game) = 0;
	virtual void endGame(float reward) = 0;
	virtual void saveAndClearReplayBuffer(ostream& os) = 0;
};

struct ReplayTurn
{
	Tensor state;
	Tensor policy;
	Tensor reward;
};

struct ReplayBuffer
{
	vector<ReplayTurn> turns;

	ReplayBuffer()
	{
		turns.reserve(1 << 24);
	}

	void save(ostream& os)
	{
		for (auto& t : turns)
		{
			t.state.serialize(os);
			t.policy.serialize(os);
			t.reward.serialize(os);
		}
	}

	void load(istream& is)
	{
		while (is.peek() != EOF)
		{
			turns.push_back(ReplayTurn());
			turns.back().state.deserialize(is);
			turns.back().policy.deserialize(is);
			turns.back().reward.deserialize(is);
		}
	}
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//											MCTS with NN implementation
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct MCTSConfig
{
	int searchIterations = 200;
	int virtualLoss = 3;
	int cPUCT = 5;
	float noiseEps = 0.25f;
	float dirichletAlpha = 0.03f;
	uint32_t changeTauTurn = 10;
};

struct MCTSModel;

struct GlobalConfig
{
	MCTSConfig mctsConfig;

	function<unique_ptr<MCTSModel>(const Game*)> buildModelFn;
	const Game* game;

	string candidateModelsPath = "./output/models/candidates";
	string bestModelsPath = "./output/models/best";

	string replayOutputPath = "./output/replays";
	uint32_t gamesPerReplayFile = 100;

	string logPath = "./output/logs";

	uint32_t gamesPerEvaluation = 200;
	float replaceRate = 0.55f;

	uint32_t gamesPerValidation = 200;

	struct OptimizerConfig
	{
		uint32_t samplesCount = 10000;
		uint32_t iterationCount = 500000;
		uint32_t batchSize = 8;
	} optimizerConfig;
};

struct MCTSModel
{
	unique_ptr<Model>	model;
	Layer*				policyOutput;
	Layer*				valueOutput;

	bool loadBest(GlobalConfig& config, fs::path& loadedFile = fs::path())
	{
		auto sortedFiles = getDateDescendingSortedFiles(config.bestModelsPath);
		if (sortedFiles.empty())
			return false;
		for (const auto& file : sortedFiles)
		{
			model->load(file);
			loadedFile = file;
			return true;
		}
		return false;
	}

	void saveAsBest(const GlobalConfig& config) const
	{
		fs::path p = config.bestModelsPath;
		checkCreateDir(p);
		p /= dateTimeNow();
		p += "-best.model";
		model->save(p);
	}

	fs::path saveAsCandidate(const GlobalConfig& config) const
	{
		fs::path p = config.candidateModelsPath;
		checkCreateDir(p);
		p /= dateTimeNow();
		p += ".model";
		model->save(p);
		return p;
	}

};

class MCTSPlayer : public Player
{
public:
	static const int MAX_ACTION_COUNT = 10;

	MCTSPlayer(const MCTSModel* model, int player, const Game& game, const MCTSConfig& config)
		: m_model(model), m_player(player), m_config(config), m_nodePoolGrowSize(1 << 12)
	{
		m_game = game.clone();
	}


	void beginGame() override
	{
		m_currentNode = nullptr;
		m_replayBufferStartIndex = static_cast<uint32_t>(m_replayBuffer.turns.size());
	}

	uint32_t chooseAction(const Game& game) override
	{
		m_actionCount = game.getActionCount(); // cache once

		if (!m_currentNode)
			m_currentNode = createNode();

		for (int i = 0; i < m_config.searchIterations; i++)
		{
			m_game->copyFrom(game);
			searchMove(m_game.get(), m_currentNode, true);
		}

		Tensor policy;
		calcPolicy(game, m_currentNode, policy);

		m_replayBuffer.turns.push_back({ game.getState(m_player), policy, Tensor({ 1,1,1 }) });

		uint32_t actionIndex = randChoice(policy);
		return actionIndex;
	}

	void notifyGameAction(uint32_t action) override
	{
		// advance on the tree and release the unused parts
		if (m_currentNode)
		{
			auto nextNode = m_currentNode->links[action].child;
			m_currentNode->links[action].child = nullptr;
			destroyNode(m_currentNode);
			m_currentNode = nextNode;
		}
	}

	void endGame(float reward) override
	{
		// write rewards
		while (m_replayBufferStartIndex < m_replayBuffer.turns.size())
		{
			m_replayBuffer.turns[m_replayBufferStartIndex++].reward[0] = reward;
		}

		// release the remains of the tree
		if (m_currentNode)
		{
			destroyNode(m_currentNode);
			m_currentNode = nullptr;
		}

		int x = 0;
		for (auto v : m_allocatedNodePools)
		{
			x += v.size();
		}
		assert(m_nodePool.size() == x);
	}

	void saveAndClearReplayBuffer(ostream& os)
	{
		assert(m_replayBufferStartIndex == m_replayBuffer.turns.size());
		m_replayBuffer.save(os);
		m_replayBuffer.turns.clear();
		m_replayBufferStartIndex = 0;
	}

private:
	struct Node;

	struct Link
	{
		int n;
		float w, q, u, p;
		Node* child;
	};

	struct Node
	{
		Node() : isLeaf(true), links{ 0 } {}
		bool isLeaf;
		Link links[MCTSPlayer::MAX_ACTION_COUNT];
	};

	void calcPolicy(const Game& game, const Node* node, Tensor& outPolicy) const
	{
		// sum(N(s,b)) for all b
		float nsum = 0.f;
		for (uint32_t i = 0; i < m_actionCount; i++)
		{
			nsum += node->links[i].n;
		}
		nsum = max<float>(nsum, 1.f);

		// normalize to be probability distribution
		outPolicy.init({ 1, 1, m_actionCount });
		for (uint32_t i = 0; i < m_actionCount; i++)
		{
			outPolicy[i] = node->links[i].n / nsum;
		}

		// if far in game (after tau turn) max out policy to (0, ..., 0, 1, 0, ..., 0) form
		if (game.getTurn() >= m_config.changeTauTurn)
		{
			uint32_t maxActionIndex = argmax(outPolicy);
			outPolicy.setZero();
			outPolicy[maxActionIndex] = 1.f;
		}
	}

	float searchMove(Game* game, Node* node, bool isRootNode = false)
	{
		assert(node);

		if (game->isFinished())
		{
			return game->getReward(m_player);
		}

		if (node->isLeaf)
		{
			if (m_model)
			{
				float leafV = expandWithModel(game, node);
				if (game->getCurrentPlayer() != m_player)
					leafV = -leafV;
				return leafV;
			}
			else
			{
				expandUniform(game, node);
			}
		}

		int actionIndex = selectAction(game, node, isRootNode);
		game->doAction(actionIndex);

		Link& nodeLink = node->links[actionIndex];
		nodeLink.n += m_config.virtualLoss;
		nodeLink.w -= m_config.virtualLoss;
		float leafV = searchMove(game, node->links[actionIndex].child);

		// backup update
		nodeLink.n = nodeLink.n - m_config.virtualLoss + 1;
		nodeLink.w = nodeLink.w + m_config.virtualLoss + leafV;
		nodeLink.q = nodeLink.w / nodeLink.n;
		return leafV;
	}

	float expandWithModel(const Game* game, Node* node)
	{
		const Tensor& state = game->getState(game->getCurrentPlayer());
		m_model->model->forward(state);
		float value = m_model->valueOutput->Y[0];

		for (uint32_t i = 0; i < m_actionCount; i++)
		{
			node->links[i].p = m_model->policyOutput->Y[i];
			node->links[i].child = createNode();
		}
		node->isLeaf = false;

		return value;
	}

	void expandUniform(const Game* game, Node* node)
	{
		for (uint32_t i = 0; i < m_actionCount; i++)
		{
			node->links[i].p = 1.0f / m_actionCount;
			node->links[i].child = createNode();
		}
		node->isLeaf = false;
	}

	int selectAction(const Game* game, const Node* node, bool isRootNode) const
	{
		// action selection with PUCT algorithm as in Alpha-Zero paper

		// sqrt(sum(N(s,b)) for all b
		float nsum = 0.f;
		for (uint32_t i = 0; i < m_actionCount; i++)
		{
			nsum += node->links[i].n;
		}
		nsum = std::max<float>(std::sqrt(nsum), 1);

		// legal actions to choose from
		Tensor legalActions;
		game->getLegalActions(legalActions);

		// dirichlet distribution added to prior p, used for root node only
		Tensor pDirichlet;
		if (isRootNode)
		{
			pDirichlet.initOnes({ 1, 1, m_actionCount });
			fillRandDirichlet(pDirichlet, m_config.dirichletAlpha);
		}

		// calculating V = Q + U and taking argmax(V)
		float maxV = -numeric_limits<float>::max();
		int maxIndex = -1;
		for (uint32_t i = 0; i < m_actionCount; i++)
		{
			float p = node->links[i].p;
			if (isRootNode)
				p = (1 - m_config.noiseEps) * p + m_config.noiseEps * pDirichlet[i];
			float u = m_config.cPUCT * p * nsum / (1.f + node->links[i].n);
			float enemyFlip = (game->getCurrentPlayer() == m_player) ? 1.0f : -1.0f;
			float v = (node->links[i].q * enemyFlip + u);
			if (v > maxV && legalActions[i] > 0.f)
			{
				maxV = v;
				maxIndex = i;
			}
		}

		assert(maxIndex >= 0);
		assert(legalActions[maxIndex] > 0.f);
		return maxIndex;
	}

	// Fast allocating nodes with pool allocator
	Node* createNode()
	{
		if (m_nodePool.empty())
		{
			m_allocatedNodePools.push_back(vector<Node>(m_nodePoolGrowSize));
			auto& newNodes = m_allocatedNodePools.back();
			for (uint32_t i = 0; i < newNodes.size(); i++)
			{
				m_nodePool.push_back(&newNodes[i]);
			}
			m_nodePoolGrowSize <<= 1; // next time grow bigger
		}
		Node* node = m_nodePool.back(); m_nodePool.pop_back();
		*node = Node();
		return node;
	};

	void destroyNode(Node* node)
	{
		for (int i = 0; i < MAX_ACTION_COUNT; i++)
			if (node->links[i].child)
				destroyNode(node->links[i].child);
		m_nodePool.push_back(node);
	}

	unique_ptr<Game> m_game;
	const MCTSModel* m_model;
	int m_player;
	uint32_t m_actionCount;
	Node* m_currentNode;
	MCTSConfig m_config;
	uint32_t m_replayBufferStartIndex;
	ReplayBuffer m_replayBuffer;

	vector<Node*> m_nodePool;
	list<vector<Node>> m_allocatedNodePools;
	uint32_t m_nodePoolGrowSize;
};

class Worker
{
public:
	Worker(const string& name, GlobalConfig& config) : m_config(config), m_working(false)
	{
		fs::path p = config.logPath;
		checkCreateDir(p);
		p /= name;
		p += ".log";
		m_ofs.open(p, ifstream::app);
	}

	virtual ~Worker() {}

	void start()
	{
		assert(!m_working);
		log("==================");
		log("[Starting Session]");
		m_working = true;
		m_thread = thread(&Worker::run, this);
	}

	void stop()
	{
		m_working = false;
	}

	void join()
	{
		m_thread.join();
	}

	template <typename T, typename... Ts>
	void log(T head, Ts... tail)
	{
		m_ofs << dateTimeNow() << ":\t";
		logRecursive(head, tail...);
	}

protected:
	virtual void run() = 0;

	GlobalConfig		m_config;
	atomic<bool>		m_working;

private:
	template <typename T, typename... Ts>
	void logRecursive(T head, Ts... tail)
	{
		m_ofs << head;
		logRecursive(tail...);
	}

	void logRecursive()
	{
		m_ofs << "\n";
		m_ofs.flush();
	}

	ofstream			m_ofs;
	thread				m_thread;
};

static int playNGames(Worker& worker, Game& game, Player* player0, Player* player1, uint32_t gameCount = 1)
{
	SimpleTimer timer;
	int player0Wins = 0;
	for (uint32_t i = 0; i < gameCount; i++)
	{
		player0->beginGame();
		player1->beginGame();

		game.reset();
		while (!game.isFinished())
		{
			Player* current = (game.getCurrentPlayer() == 0) ? player0 : player1;
			uint32_t action = current->chooseAction(game);
			game.doAction(action);
			player0->notifyGameAction(action);
			player1->notifyGameAction(action);
		}

		const float reward0 = game.getReward(0);
		const float reward1 = game.getReward(1);

		player0->endGame(reward0);
		player1->endGame(reward1);

		if (reward0 > reward1)
			player0Wins++;
	}

	worker.log("played ", gameCount, " games, took: ", timer.getElapsedSeconds(), "s");

	return player0Wins;
}

class SelfPlayWorker : public Worker
{
public:
	SelfPlayWorker(GlobalConfig& config) : Worker("self_play", config) {}

	void run() override
	{
		// model
		auto model = m_config.buildModelFn(m_config.game);
		auto game = m_config.game->clone();

		while (m_working)
		{
			log("starting iteration");
			if (model->loadBest(m_config))
				log("best model loaded");

			MCTSPlayer player0(model.get(), 0, *game, m_config.mctsConfig);
			MCTSPlayer player1(model.get(), 1, *game, m_config.mctsConfig);
			playNGames(*this, *game, &player0, &player1, m_config.gamesPerReplayFile);

			checkCreateDir(m_config.replayOutputPath);
			fs::path filename = fs::path(m_config.replayOutputPath) / dateTimeNow();
			filename += ".bin";
			{
				ofstream ofs(filename, ifstream::out | ios::binary);
				player0.saveAndClearReplayBuffer(ofs);
				player1.saveAndClearReplayBuffer(ofs);

				log("saved replay: ", filename);
			}

			this_thread::sleep_for(10ms);
		}
	}
};

class OptimizeWorker : public Worker
{
public:
	OptimizeWorker(GlobalConfig& config) : Worker("optimization", config) {}

	void run() override
	{
		// build model template
		auto model = m_config.buildModelFn(m_config.game);

		// init trainer
		AdamTrainer t;
		t.lr = 0.001f;
		t.batchSize = m_config.optimizerConfig.batchSize;
		t.init(model->model.get());

		while (m_working)
		{
			log("starting iteration");

			// load best model
			if (!model->loadBest(m_config))
			{
				log("no best model yet, starting with random");
			}

			// load replay buffer (load most recent files till buffer is full, remove others)
			ReplayBuffer buffer;

			while (1)
			{
				auto sortedFiles = getDateDescendingSortedFiles(m_config.replayOutputPath);
				for (const auto& file : sortedFiles)
				{
					if (buffer.turns.size() < m_config.optimizerConfig.samplesCount)
					{
						ifstream ifs(file, ifstream::in | ios::binary);
						buffer.load(ifs);
					}
					else
					{
						fs::remove(file);
					}
					log("added replays from file: ", file);
				}

				if (!buffer.turns.empty())
					break;

				this_thread::sleep_for(1s); // wait and iterate till the first file
			}

			// optimize
			log("starting optimization");
			SimpleTimer timer;
			const uint32_t iterations = m_config.optimizerConfig.iterationCount;
			for (uint32_t i = 0; i < iterations; i++)
			{
				if (i % (iterations / 10) == 0)
				{
					log("optimization loop ", i, " of ", iterations);
				}

				int sample = randUniform(0, (int)buffer.turns.size() - 1);
				t.train({ &buffer.turns[sample].state }, { &buffer.turns[sample].policy, &buffer.turns[sample].reward });
			}
			log("finished optimization, took: ", timer.getElapsedSeconds(), "s loss:", t.getLoss());

			// save model as candidate
			fs::path p = model->saveAsCandidate(m_config);
			log("model saved: ", p);

			this_thread::sleep_for(100ms);
		}
	}
};

class EvaluateWorker : public Worker
{
public:
	EvaluateWorker(GlobalConfig& config) : Worker("evaluation", config) {}

	void run() override
	{
		auto game = m_config.game->clone();

		while (m_working)
		{
			//log("starting iteration");

			// build and load best model
			auto bestModel = m_config.buildModelFn(m_config.game);
			if (!bestModel->loadBest(m_config))
			{
				log("no best model yet, evaluating against random model");
			}

			auto candidateModel = m_config.buildModelFn(m_config.game);
			auto sortedFiles = getDateDescendingSortedFiles(m_config.candidateModelsPath);
			for (const auto& file : sortedFiles)
			{
				candidateModel->model->load(file);

				MCTSPlayer player0(candidateModel.get(), 0, *game, m_config.mctsConfig);
				MCTSPlayer player1(bestModel.get(), 1, *game, m_config.mctsConfig);
				int player0Wins = playNGames(*this, *game, &player0, &player1, m_config.gamesPerEvaluation);

				float winRate = player0Wins / static_cast<float>(m_config.gamesPerEvaluation);
				if (winRate >= m_config.replaceRate)
				{
					log("evaluated (new best): ", file, " win rate:", winRate);
					candidateModel->saveAsBest(m_config);
				}
				else
				{
					log("evaluated (fail): ", file, " win rate:", winRate);
				}

				error_code ec;
				if (!fs::remove(file, ec))
				{
					log("remove error: ", ec);
				}
				break;
			}

			if (sortedFiles.empty())
			{
				log("no candidates to evaluate, waiting");
				this_thread::sleep_for(5s);
			}
		}
	}
};

class ValidationWorker : public Worker
{
public:
	ValidationWorker(GlobalConfig& config) : Worker("validation", config) {}

	void run() override
	{
		auto game = m_config.game->clone();

		fs::path lastBestModel;

		while (m_working)
		{
			//log("starting iteration");

			// build and load best model
			auto bestModel = m_config.buildModelFn(m_config.game);
			fs::path bestModelFile;
			if (!bestModel->loadBest(m_config, bestModelFile) || bestModelFile == lastBestModel)
			{
				//log("no best model yet");
				this_thread::sleep_for(10s);
				continue;
			}
			lastBestModel = bestModelFile;

			const vector<int> iterations = { 1000, 10000, 20000 }; // TODO: config
			for (auto iters : iterations)
			{
				MCTSPlayer player0(bestModel.get(), 0, *game, m_config.mctsConfig);
				MCTSConfig mctsCustom;
				mctsCustom.searchIterations = iters;
				MCTSPlayer player1(nullptr, 1, *game, mctsCustom);
				int player0Wins = playNGames(*this, *game, &player0, &player1, m_config.gamesPerValidation);

				float winRate = player0Wins / static_cast<float>(m_config.gamesPerEvaluation);
				log("validation result: ", bestModelFile, " iters: ", iters, " win rate:", winRate);
			}
		}
	}
};