#pragma once

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