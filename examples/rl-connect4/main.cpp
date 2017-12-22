#include <ai.h>
#include <numeric>
#define LOG_TO_FILE 1

template <typename T>
vector<size_t> sortIndexes(const vector<T> &v) 
{
	// initialize original index locations
	vector<size_t> idx(v.size());
	iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });

	return idx;
}

class Game
{
public:
	virtual uint32_t getPlayerCount() const = 0;
	virtual uint32_t getActionCount() const = 0;
	virtual Dim getStateDim() const = 0;
	virtual uint32_t getCurrentPlayer() const = 0;
	virtual bool isFinished() const = 0;
	virtual float getReward(uint32_t player) const = 0;
	virtual const Tensor& getState(uint32_t player) const = 0;
	virtual void doAction(const Tensor& action, uint32_t player) = 0;
	virtual void reset() = 0;
	virtual void render() {}
};

class Connect4 : public Game
{
public:
	uint32_t getPlayerCount() const override { return 2; }

	uint32_t getActionCount() const override { return 7; }

	Dim getStateDim() const override { return Dim{ 7,6,1 }; }
	
	uint32_t getCurrentPlayer() const override { return m_currentPlayer; }

	bool isFinished() const override { return !m_running; }

	float getReward(uint32_t player) const override { return m_running ? 0.f : (m_winner == player ? 1.f : -1.f); }

	const Tensor& getState(uint32_t player) const
	{
		return player == 0 ? m_stateP0 : m_stateP1;
	}

	void doAction(const Tensor& action, uint32_t player) override
	{
		assert(!isFinished());

		const auto sortedActions = sortIndexes(action.data);
		for (int i = sortedActions.size() - 1; i >= 0; i--)
		{
			int x = sortedActions[i];
			const uint32_t maxY = getStateDim().sy;
			for (uint32_t y = 0; y < maxY; y++)
			{
				if (tryPlaceOn(x, y))
				{
					return;
				}
			}
		}
		assert(false);
	}

	void reset() override
	{
		m_currentPlayer = randBernoulli() ? 1 : 0;
		m_winner = -1;

		m_stateP0.initZero(getStateDim());
		m_stateP1.initZero(getStateDim());
		m_running = true;
		totalMoves = 0;
	}

	void render() override
	{
		std::cout << "=========================================================\n";
		std::cout << "Game is: " << (isFinished() ? "FINISHED" : "RUNNING") << "\n";
		std::cout << "Winner: " << m_winner<< "\n";
		std::cout << "Current player: " << m_currentPlayer << "\n";
		std::cout << "Total moves: " << totalMoves << "\n";
		for (uint32_t y = 0; y < m_stateP0.dim.sy; y++)
		{
			for (uint32_t x = 0; x < m_stateP0.dim.sx; x++)
			{
				int v = getBoardValue(x, y);
				std::cout << (v == -1 ? '.' : (v == 1 ? 'x' : 'o'));
			}
			std::cout << "\n";
		}
		//std::cout << "State p0:" << m_stateP0;
		//std::cout << "State p1:" << m_stateP1;
		std::cout << "=========================================================\n";
	}

	Connect4()
	{
		reset();
	}
private:
	int getBoardValue(int x, int y)
	{
		if (m_stateP0.get(x, y, 0) > 0.f)
			return 0;
		else if (m_stateP0.get(x, y, 0) < 0.f)
			return 1;
		else
			return -1;
	}

	bool tryPlaceOn(int x, int y)
	{
		if (getBoardValue(x, y) == -1)
		{
			m_stateP0.set(x, y, 0, m_currentPlayer == 0 ? 1.f : -1.f);
			m_stateP1.set(x, y, 0, m_stateP0.get(x, y, 0) * -1.f);

			const int sx = getStateDim().sx;
			const int sy = getStateDim().sy;
			// check end
			int sums[4] = { 0 };
			for (int dx = -1; dx <= 1; dx++)
			{
				for (int dy = -1; dy <= 1; dy++)
				{
					if(dx == 0 && dy == 0)
						continue;
					int group = dx * dy + (1 - dx * dx * dy * dy)*(2 * dx * dx) + 1;
					for (int d = 1; d <= 3; d++)
					{
						int x1 = x + d * dx;
						int y1 = y + d * dy;
						if (x1 >= 0 && x1 < sx && y1 >= 0 && y1 < sy && getBoardValue(x1, y1) == m_currentPlayer)
						{
							sums[group]++;
						}
						else
						{
							break;
						}
					}
				}
			}
			for(int i = 0; i < 4; i++)
			{
				if (sums[i] >= 3)
				{
					m_running = false;
					m_winner = m_currentPlayer;
				}
			}

			totalMoves++;

			if (m_running && totalMoves >= 6 * 7)
			{
				m_running = false;
				m_winner = -1;
			}

			m_currentPlayer = (m_currentPlayer + 1) % 2;

			return true;
		}
		return false;
	}

	int m_currentPlayer;
	bool m_running;
	int m_winner;
	Tensor m_stateP0;
	Tensor m_stateP1;
	uint32_t totalMoves;
};


int main()
{
#if LOG_TO_FILE
	freopen("../output.txt", "w", stdout);
#endif

	Connect4 game;

	for (int i = 0; i < 200; i++)
	{
		std::cout << "NEW GAME\n\n";

		game.reset();
		game.render();
		while (!game.isFinished())
		{
			int player = game.getCurrentPlayer();
			Tensor action({ 1, 1, game.getActionCount() });
			action.setRand();
			game.doAction(action, player);
			std::cout << "p0 reward: " << game.getReward(0) << "\n";
			std::cout << "p1 reward: " << game.getReward(1) << "\n";
			game.render();
		}
	}

#if 0
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

	const float epochs = 50;
	for (int i = 0; i < epochs; i++)
	{
		t.train(X, Y);
		std::cout << "Loss:" << t.getLoss() << "\n";

	}
	model.forward(X);
	std::cout << output->Y;
#endif

	//getchar();

	return 0;
}