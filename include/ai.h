#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <cassert>
#include <iostream>
#include <chrono>
#include <future>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//														Common
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::default_random_engine g_randomGen;

struct Dim
{
	uint32_t sx;
	uint32_t sy;
	uint32_t depth;

	uint32_t size() const
	{
		return sx * sy * depth;
	}
};

struct Tensor
{
	void init(Dim d)
	{
		dim = d;
		data.resize(dim.size(), 0);
	}

	void initRand(Dim d)
	{
		dim = d;
		data.resize(dim.size());
		const float scale = std::sqrt(1.0f / dim.size());
		std::normal_distribution<double> distribution(0.f, scale);
		for (auto& x : data)
		{
			x = distribution(g_randomGen);
		}
	}

	void setZero()
	{
		std::fill(data.begin(), data.end(), 0);
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
		for (int d = 0; d < t.dim.depth; d++)
		{
			if (t.dim.depth > 1)
				os << "depth:" << d << "\n";
			for (int y = 0; y < t.dim.sy; y++)
			{
				for (int x = 0; x < t.dim.sx; x++)
				{
					os << t.get(x, y, d) << " ";
				}
				os << "\n";
			}
		}
		return os;
	}

	Dim dim;
	std::vector<float> data;
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//														Layers
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct OptimizationData
{
	std::vector<float>& theta;
	std::vector<float>& dtheta;
	float l1;
	float l2;
};

class Layer
{
public:
	Dim outDim;
	Tensor outAct;
	Tensor outGrad;

	virtual void initialize(Dim inDim) = 0;
	virtual void forward(const Tensor& inAct) = 0;
	virtual void backward(const Tensor& inAct, const Tensor& inGrad) = 0;
	virtual float calcLoss(const Tensor& y) { return 0.f; }
	virtual void fillOptimizationData(std::vector<OptimizationData>& data) {};

	virtual ~Layer() {}
};

class DotLayer : public Layer
{
public:
	float l1 = 0.f;
	float l2 = 1.f;

	void fillOptimizationData(std::vector<OptimizationData>& data) override
	{
		for (uint32_t i = 0; i < m_filters.size(); i++)
		{
			data.push_back(OptimizationData{ m_filters[i].data, m_filtersGrad[i].data, l1, l2 });
		}
		data.push_back(OptimizationData{ m_biases.data, m_biasesGrad.data, 0.f, 0.f });
	}

protected:
	std::vector<Tensor> m_filters;
	std::vector<Tensor> m_filtersGrad;
	Tensor m_biases;
	Tensor m_biasesGrad;
};

class Dense : public DotLayer
{
public:
	Dense(uint32_t n)
	{
		outDim = { 1, 1, n };

		outAct.init({ 1, 1, outDim.depth });
		m_filters.resize(n);
		m_filtersGrad.resize(n);
		m_biases.initRand({ 1, 1, n });
		m_biasesGrad.init({ 1, 1, n });
	}

	void initialize(Dim inDim) override
	{
		outGrad.init(inDim);
		for (auto& f : m_filters)
			f.initRand({ 1, 1, inDim.size() });
		for (auto& f : m_filtersGrad)
			f.init({ 1, 1, inDim.size() });
	}

	void forward(const Tensor& inAct) override
	{
		for (uint32_t i = 0; i < outDim.depth; i++)
		{
			float a = 0.f;
			const auto& f = m_filters[i];
			for (uint32_t k = 0; k < f.dim.depth; k++)
			{
				a += inAct.data[k] * f.data[k];
			}
			a += m_biases.data[i];
			outAct.data[i] = a;
		}
	}

	void backward(const Tensor& inAct, const Tensor& inGrad) override
	{
		outGrad.setZero();
		for (uint32_t i = 0; i < outDim.depth; i++)
		{
			const auto& f = m_filters[i];
			auto& fg = m_filtersGrad[i];
			for (uint32_t k = 0; k < f.dim.depth; k++)
			{
				outGrad.data[k] += f.data[k] * inGrad.data[i];
				fg.data[k] += inAct.data[k] * inGrad.data[i];
			}
			m_biasesGrad.data[i] += inGrad.data[i];
		}
	}
};


class Conv : public DotLayer
{
public:
	Conv(uint32_t filters, uint32_t kernel_x, uint32_t kernel_y, uint32_t stride, uint32_t pad)
	{
		outDim.depth = filters;
		m_ksx = kernel_x;
		m_ksy = kernel_y;
		m_stride = stride;
		m_pad = pad;

		m_filters.resize(outDim.depth);
		m_filtersGrad.resize(outDim.depth);
		m_biases.initRand({ 1, 1, outDim.depth });
		m_biasesGrad.init({ 1, 1, outDim.depth });
	}

	void initialize(Dim inDim) override
	{
		outDim.sx = (uint32_t)std::floor(((float)inDim.sx + m_pad * 2 - m_ksx) / m_stride + 1);
		outDim.sy = (uint32_t)std::floor(((float)inDim.sy + m_pad * 2 - m_ksy) / m_stride + 1);
		outAct.init(outDim);
		outGrad.init(inDim);

		for (auto& f : m_filters)
			f.initRand({ m_ksx, m_ksy, inDim.depth });
		for (auto& f : m_filtersGrad)
			f.init({ m_ksx, m_ksy, inDim.depth });
	}

	void forward(const Tensor& inAct) override
	{
		const int in_sx = inAct.dim.sx;
		const int in_sy = inAct.dim.sy;
		for (uint32_t d = 0; d < outDim.depth; d++)
		{
			const Tensor& f = m_filters[d];
			int y = -m_pad;
			for (uint32_t ay = 0; ay < outDim.sy; ay++)
			{
				int x = -m_pad;
				for (uint32_t ax = 0; ax < outDim.sx; ax++)
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
									a += f.get(fx, fy, fd) * inAct.get(ox, oy, fd);
								}
							}

						}
					}

					a += m_biases.data[d];
					outAct.set(ax, ay, d, a);

					x += m_stride;
				}
				y += m_stride;
			}
		}
	}

	void backward(const Tensor& inAct, const Tensor& inGrad) override
	{
		outGrad.setZero();

		const int in_sx = inAct.dim.sx;
		const int in_sy = inAct.dim.sy;
		for (uint32_t d = 0; d < outDim.depth; d++)
		{
			const Tensor& f = m_filters[d];
			Tensor& fg = m_filtersGrad[d];
			int y = -m_pad;
			for (uint32_t ay = 0; ay < outDim.sy; ay++)
			{
				int x = -m_pad;
				for (uint32_t ax = 0; ax < outDim.sx; ax++)
				{
					float chain_grad = inGrad.get(ax, ay, d);
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
									fg.add(fx, fy, fd, (inAct.get(ox, oy, fd) * chain_grad));
									outGrad.add(ox, oy, fd, (f.get(fx, fy, fd) * chain_grad));
								}
							}
						}
					}
					m_biasesGrad.data[d] += chain_grad;

					x += m_stride;
				}
				y += m_stride;
			}
		}
	}

private:
	uint32_t m_ksx;
	uint32_t m_ksy;
	uint32_t m_stride;
	int m_pad;
};

class Relu : public Layer
{
public:
	void initialize(Dim inDim) override
	{
		outDim = inDim;
		outAct.init(outDim);
		outGrad.init(inDim);
	}

	void forward(const Tensor& inAct) override
	{
		for (uint32_t i = 0; i < outDim.size(); i++)
		{
			outAct.data[i] = inAct.data[i] < 0 ? 0 : inAct.data[i];
		}
	}

	void backward(const Tensor& inAct, const Tensor& inGrad) override
	{
		for (uint32_t i = 0; i < outDim.size(); i++)
		{
			outGrad.data[i] = outAct.data[i] <= 0 ? 0 : inGrad.data[i];
		}
	}
};

class Softmax : public Layer
{
public:
	Softmax()
	{
	}

	void initialize(Dim inDim) override
	{
		outDim = { 1, 1, inDim.size() };
		outAct.init(outDim);
		outGrad.init(inDim);
	}

	void forward(const Tensor& inAct) override
	{
		// max act
		float amax = *std::max_element(inAct.data.begin(), inAct.data.end());

		// compute exp
		float esum = 0.f;
		for (uint32_t i = 0; i < outDim.depth; i++)
		{
			const float e = std::exp(inAct.data[i] - amax);
			esum += e;
			outAct.data[i] = e;
		}

		// norm
		for (uint32_t i = 0; i < outDim.depth; i++)
		{
			outAct.data[i] /= esum;
		}
	}

	void backward(const Tensor& inAct, const Tensor& inGrad) override
	{
		uint32_t index = std::distance(inGrad.data.begin(), std::max_element(inGrad.data.begin(), inGrad.data.end()));
		for (uint32_t i = 0; i < outDim.depth; i++)
		{
			float indicator = (index == i) ? 1.f : 0.f;
			outGrad.data[i] = -(indicator - outAct.data[i]);
		}
	}

	float calcLoss(const Tensor& y) override
	{
		uint32_t index = std::distance(y.data.begin(), std::max_element(y.data.begin(), y.data.end()));
		return (-std::log(outAct.data[index])); // not sure here
	}
};

class Regression : public Layer
{
public:
	Regression()
	{
	}

	void initialize(Dim inDim) override
	{
		outDim = { 1, 1, inDim.size() };
		outAct.init(outDim);
		outGrad.init(inDim);
	}

	void forward(const Tensor& inAct) override
	{
		for (uint32_t i = 0; i < outDim.depth; i++)
		{
			outAct.data[i] = inAct.data[i];
		}
	}

	void backward(const Tensor& inAct, const Tensor& inGrad) override
	{
		const float factor = 2.f / outDim.depth;
		for (uint32_t i = 0; i < outDim.depth; i++)
		{
			float dy = (outAct.data[i] - inGrad.data[i]);
			outGrad.data[i] = factor * dy;
		}
	}

	float calcLoss(const Tensor& y) override
	{
		float loss = 0.f;
		for (uint32_t i = 0; i < outDim.depth; i++)
		{
			float dy = (outAct.data[i] - y.data[i]);
			loss += dy * dy;
		}
		loss /= outDim.depth;
		return loss;
	}
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//													Neural Net
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class NN
{
public:

	~NN()
	{
		for (auto& f : m_layers)
		{
			delete f;
		}
	}

	void initialize(Dim inDim)
	{
		Dim prevDim = inDim;
		for (uint32_t i = 0; i < m_layers.size(); i++)
		{
			m_layers[i]->initialize(prevDim);
			prevDim = m_layers[i]->outDim;
		}
	}

	const Tensor& forward(const Tensor& x)
	{
		const Tensor* inAct = &x;
		for (uint32_t i = 0; i < m_layers.size(); i++)
		{
			m_layers[i]->forward(*inAct);
			inAct = &m_layers[i]->outAct;
		}
		return *inAct;
	}

	float forward(const Tensor& x, const Tensor& y)
	{
		forward(x);
		return m_layers.back()->calcLoss(y); // assume last
	}

	const Tensor& backward(const Tensor& x, const Tensor& y)
	{
		const Tensor* inGrad = &y;
		for (int i = m_layers.size() - 1; i >= 0; i--)
		{
			const Tensor* inAct = (i > 0 ? &m_layers[i - 1]->outAct : &x);
			m_layers[i]->backward(*inAct, *inGrad);
			inGrad = &m_layers[i]->outGrad;
		}
		return *inGrad;
	}

	void addDense(uint32_t n)
	{
		m_layers.push_back(new Dense(n));
	}

	void addConv(uint32_t filters, uint32_t kernel_x, uint32_t kernel_y, uint32_t stride = 1, uint32_t pad = 0)
	{
		m_layers.push_back(new Conv(filters, kernel_x, kernel_y, stride, pad));
	}

	void AddRelu()
	{
		m_layers.push_back(new Relu());
	}

	void addSoftmax()
	{
		m_layers.push_back(new Softmax());
	}

	void addRegression()
	{
		m_layers.push_back(new Regression());
	}

	void fillOptimizationData(std::vector<OptimizationData>& data)
	{
		for (auto& l : m_layers)
		{
			l->fillOptimizationData(data);
		}
	}

private:
	std::vector<Layer*> m_layers;
};

std::ostream& operator<<(std::ostream& os, const std::vector<float>& v)
{
	os << "[";
	for (auto x : v)
	{
		os << x << ",";
	}
	os << "]\n";
	return os;
}


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

	void init(NN* net)
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

	void train(const Tensor& x, const Tensor& y)
	{
		float cost_loss = m_net->forward(x, y);
		m_net->backward(x, y);
		m_lossAcc += cost_loss;

		m_iter++;
		if (m_iter % batchSize == 0)
		{
			std::vector<std::future<float>> futures;
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
	NN* m_net;
	std::vector<OptimizationData> m_optData;
	std::vector<std::vector<OptimizerT>> m_kernels;
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
		const float biasCorr1 = m * (1.f - std::pow<float>(s.beta1, t->m_iter));
		const float biasCorr2 = v * (1.f - std::pow<float>(s.beta2, t->m_iter));
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