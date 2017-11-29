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

	float& operator[](size_t idx) 
	{ 
		return data[idx]; 
	}
	
	const float& operator[](size_t idx) const 
	{ 
		return data[idx]; 
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
	Tensor Y;
	Tensor dX;

	virtual void initialize(Dim inDim) = 0;
	virtual void forward(const Tensor& X) = 0;
	virtual void backward(const Tensor& X, const Tensor& dY) = 0;
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
		Y.init({ 1, 1, n });
		m_filters.resize(n);
		m_filtersGrad.resize(n);
		m_biases.initRand({ 1, 1, n });
		m_biasesGrad.init({ 1, 1, n });
	}

	void initialize(Dim inDim) override
	{
		dX.init(inDim);
		for (auto& f : m_filters)
			f.initRand({ 1, 1, inDim.size() });
		for (auto& f : m_filtersGrad)
			f.init({ 1, 1, inDim.size() });
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
	Conv(uint32_t filters, uint32_t kernel_x, uint32_t kernel_y, uint32_t stride, uint32_t pad)
	{
		m_filterCount = filters;
		m_ksx = kernel_x;
		m_ksy = kernel_y;
		m_stride = stride;
		m_pad = pad;
	}

	void initialize(Dim inDim) override
	{
		Dim outDim;
		outDim.depth = m_filterCount;
		outDim.sx = (uint32_t)std::floor(((float)inDim.sx + m_pad * 2 - m_ksx) / m_stride + 1);
		outDim.sy = (uint32_t)std::floor(((float)inDim.sy + m_pad * 2 - m_ksy) / m_stride + 1);

		Y.init(outDim);
		dX.init(inDim);

		m_filters.resize(outDim.depth);
		m_filtersGrad.resize(outDim.depth);
		m_biases.initRand({ 1, 1, outDim.depth });
		m_biasesGrad.init({ 1, 1, outDim.depth });

		for (auto& f : m_filters)
			f.initRand({ m_ksx, m_ksy, inDim.depth });
		for (auto& f : m_filtersGrad)
			f.init({ m_ksx, m_ksy, inDim.depth });
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
	uint32_t m_filterCount;
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
		Y.init(inDim);
		dX.init(inDim);
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

class Softmax : public Layer
{
public:
	Softmax()
	{
	}

	void initialize(Dim inDim) override
	{
		Y.init(inDim);
		dX.init(inDim);
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

struct Loss
{
	virtual float f(const Tensor& X, const Tensor& Y) const = 0;
	virtual void df(Tensor& dX, const Tensor& X, const Tensor& Y) const = 0;
};

struct MSE : public Loss
{
	virtual float f(const Tensor& X, const Tensor& Y) const override
	{
		float loss = 0.f;
		const uint32_t n = X.dim.size();
		for (uint32_t i = 0; i < n; i++)
		{
			float dy = (X[i] - Y[i]);
			loss += dy * dy;
		}
		loss /= n;
	
		return loss;
	}

	virtual void df(Tensor& dX, const Tensor& X, const Tensor& Y) const override
	{
		const uint32_t n = X.dim.size();
		const float factor = 2.f / n;
		for (uint32_t i = 0; i < n; i++)
		{
			float dy = (X[i] - Y[i]);
			dX[i] = factor * dy;
		}
	}

};

struct CrossEntropy : public Loss
{
	virtual float f(const Tensor& X, const Tensor& Y) const override
	{
		float loss = 0.f;
		const uint32_t n = X.dim.size();
		for (uint32_t i = 0; i < n; i++)
		{
			loss += -Y[i] * std::log(X[i]) - (1.f - Y[i]) * std::log((1.f - X[i]));
		}

		return loss;
	}

	virtual void df(Tensor& dX, const Tensor& X, const Tensor& Y) const override
	{
		const uint32_t n = X.dim.size();
		for (uint32_t i = 0; i < n; i++)
		{
			dX[i] = (X[i] - Y[i]) / (X[i] * (1.f - X[i]));
		}
	}

};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//													Neural Net
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class NN
{
public:
	NN() : m_loss(std::make_shared<MSE>())
	{
	}

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
			prevDim = m_layers[i]->Y.dim;
		}
	}

	const Tensor& forward(const Tensor& X)
	{
		const Tensor* cur_X = &X;
		for (uint32_t i = 0; i < m_layers.size(); i++)
		{
			m_layers[i]->forward(*cur_X);
			cur_X = &m_layers[i]->Y;
		}
		return *cur_X;
	}

	float forward(const Tensor& X, const Tensor& Y)
	{
		const Tensor& act = forward(X);
		return m_loss->f(act, Y);
	}

	const Tensor& backward(const Tensor& X, const Tensor& Y)
	{
		assert(m_layers.size());
		const auto lastLayer = m_layers.back();
		m_lossGrad.init(lastLayer->Y.dim);
		m_loss->df(m_lossGrad, lastLayer->Y, Y);
		const Tensor* cur_dX = &m_lossGrad;
		for (int i = m_layers.size() - 1; i >= 0; i--)
		{
			const Tensor* cur_X = (i > 0 ? &m_layers[i - 1]->Y : &X);
			m_layers[i]->backward(*cur_X, *cur_dX);
			cur_dX = &m_layers[i]->dX;
		}
		return *cur_dX;
	}

	void addLayer(std::shared_ptr<Layer> layer)
	{

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

	void setLoss(std::shared_ptr<Loss> loss)
	{
		m_loss = loss;
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
	std::shared_ptr<Loss> m_loss;
	Tensor				  m_lossGrad;
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

	void train(const Tensor& X, const Tensor& Y)
	{
		float cost_loss = m_net->forward(X, Y);
		m_net->backward(X, Y);
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