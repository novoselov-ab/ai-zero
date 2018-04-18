#include "ai.h"
#define LOG_TO_FILE 1


// Connect4 Game implementation:
class Connect4 : public Game
{
public:
	uint32_t getPlayerCount() const override { return 2; }

	uint32_t getActionCount() const override { return 7; }

	Dim getStateDim() const override { return Dim{ 7,6,2 }; }
	
	uint32_t getCurrentPlayer() const override { return m_currentPlayer; }

	uint32_t getTurn() const override { return m_turn; }

	bool isFinished() const override { return !m_running; }

	float getReward(uint32_t player) const override { return m_running ? 0.f : (m_winner == player ? 1.f : -1.f); }

	const Tensor& getState(uint32_t player) const
	{
		return player == 0 ? m_stateP0 : m_stateP1;
	}

	void getLegalActions(Tensor& legalActions) const override
	{
		legalActions.initZero({ 1, 1, getActionCount() });
		const Dim& d = getStateDim();
		for (uint32_t x = 0; x < d.sx; x++)
		{
			if (getBoardValue(x, d.sy - 1) == -1)
				legalActions[x] = 1;
		}
	}

	void doAction(const Tensor& action) override
	{
		assert(!isFinished());

		uint32_t actionIndex = argmax(action);
		const uint32_t maxY = getStateDim().sy;
		for (uint32_t y = 0; y < maxY; y++)
		{
			if (tryPlaceOn(actionIndex, y))
			{
				return;
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
		m_turn = 0;
	}

	unique_ptr<Game> clone() const override
	{
		return make_unique<Connect4>(*this);
	}

	void copyFrom(const Game& other) override
	{
		*this = static_cast<const Connect4&>(other);
	}


	void render() override
	{
		std::cout << "=========================================================\n";
		std::cout << "Game is: " << (isFinished() ? "FINISHED" : "RUNNING") << "\n";
		std::cout << "Winner: " << m_winner<< "\n";
		std::cout << "Current player: " << m_currentPlayer << "\n";
		std::cout << "Total moves: " << m_turn << "\n";
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
		std::cout << "=========================================================" << std::endl;
	}

	Connect4()
	{
		reset();
	}
private:
	int getBoardValue(int x, int y) const
	{
		if (m_stateP0.get(x, y, 0) > 0.f)
			return 0;
		else if (m_stateP0.get(x, y, 1) > 0.f)
			return 1;
		else
			return -1;
	}

	bool tryPlaceOn(int x, int y)
	{
		if (getBoardValue(x, y) == -1)
		{
			m_stateP0.set(x, y, m_currentPlayer, 1.f);
			m_stateP1.set(x, y, (m_currentPlayer + 1) % 2, 1.f);

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

			m_turn++;

			if (m_running && m_turn >= 6 * 7)
			{
				m_running = false;
				m_winner = -1;
			}

			m_currentPlayer = (m_currentPlayer + 1) % 2;

			return true;
		}
		return false;
	}

	int			m_currentPlayer;
	bool		m_running;
	int			m_winner;
	Tensor		m_stateP0;
	Tensor		m_stateP1;
	uint32_t	m_turn;
};

// Model used for training
static unique_ptr<MCTSModel> buildModel1(const Game* game)
{
	auto mctsModel = make_unique<MCTSModel>();

	// model
	Dim inputDim = game->getStateDim();
	auto input = make_shared<Input>(inputDim);
	auto x = (*make_shared<Conv>(16, 3, 3, 2, 1))(input);
	x = (*make_shared<Relu>())(x);
	//x = (*make_shared<Conv>(16, 3, 3, 2, 1))(x);
	//x = (*make_shared<Relu>())(x);
	auto split = (*make_shared<Dense>(10))(x);
	auto px = (*make_shared<Dense>(10))(split);
	px = (*make_shared<Dense>(game->getActionCount()))(px);
	auto policyOutput = make_shared<Softmax>();
	auto policyLoss = make_shared<CrossEntropy>();
	px = (*policyOutput)(px);
	px = (*policyLoss)(px);

	auto vx = (*make_shared<Dense>(10))(split);
	auto valueOutput = (*make_shared<Dense>(1))(vx);
	auto valueLoss = make_shared<MSE>();
	x = (*valueLoss)(valueOutput);

	vector<shared_ptr<LossLayer>> losses = { policyLoss, valueLoss };
	vector<shared_ptr<Input>> inputs = { input };
	mctsModel->model = make_unique<Model>(inputs, losses);
	mctsModel->policyOutput = policyOutput.get();
	mctsModel->valueOutput = valueOutput.get();
	return mctsModel;
}

int main()
{
#if LOG_TO_FILE
	freopen("../output.txt", "w", stdout);
	freopen("../error.txt", "w", stderr);
#endif

#if 0
	Connect4 game;

	for (int i = 0; i < 200; i++)
	{
		std::cout << "NEW GAME\n\n";

		game.reset();
		game.render();
		while (!game.isFinished())
		{
			//int player = game.getCurrentPlayer();
			Tensor action({ 1, 1, game.getActionCount() });
			action.setRand();
			game.doAction(action);
			std::cout << "p0 reward: " << game.getReward(0) << "\n";
			std::cout << "p1 reward: " << game.getReward(1) << "\n";
			game.render();
		}
	}
#endif

	// Actual main training loop with all 4 workers running in parallel
	Connect4 game;

	GlobalConfig config;
	config.buildModelFn = buildModel1;
	config.game = &game;

	{
		SelfPlayWorker selfPlay(config);
		OptimizeWorker optimizer(config);
		EvaluateWorker evaluate(config);
		ValidationWorker validate(config);
		vector<Worker*> workers = { &selfPlay, &optimizer, &evaluate, &validate };

		for (auto w : workers) w->start();
		
		while(1)
		{
			if (getchar() == 'q')
				break;
		}

		for (auto w : workers) w->stop();
		for (auto w : workers) w->join();
	}

	return 0;
}