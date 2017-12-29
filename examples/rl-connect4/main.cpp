#include <ai.h>
#include <numeric>
#define LOG_TO_FILE 1

#if 0
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
#endif

class Game
{
public:
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

	int m_currentPlayer;
	bool m_running;
	int m_winner;
	Tensor m_stateP0;
	Tensor m_stateP1;
	uint32_t m_turn;
};

class Player
{
public:
	virtual void beginGame() = 0;
	virtual void notifyGameAction(uint32_t action) = 0;
	virtual uint32_t chooseAction(const Game& game) = 0;
	virtual void endGame(float reward) = 0;
	virtual void saveAndClearReplayBuffer(ostream& os) = 0;
};

struct MCTSModel
{
	unique_ptr<Model> model;
	Layer* policyOutput;
	Layer* valueOutput;
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
		while(is.peek() != EOF)
		{
			turns.push_back(ReplayTurn());
			turns.back().state.deserialize(is);
			turns.back().policy.deserialize(is);
			turns.back().reward.deserialize(is);
}
	}
};

struct MCTSConfig
{
	int searchIterations = 10;
	int virtualLoss = 3;
	int cPUCT = 5;
	float noiseEps = 0.25f;
	float dirichletAlpha = 0.03f;
	uint32_t changeTauTurn = 10;
};

struct OptimizerConfig
{
	uint32_t samplesCount = 1000;
	uint32_t iterationCount = 10;
	uint32_t batchSize = 8;
};

struct GlobalConfig
{
	MCTSConfig mctsConfig;
	function<unique_ptr<MCTSModel>(const Game*)> buildModelFn;
	const Game* game;

	string modelOutputPath = "./models";
	string bestModelFilename = "best.model";

	string replayOutputPath = "./replays";
	string replayFilePrefix = "replays_";
	uint32_t gamesPerReplayFile = 10;

	string logPath = "./logs";

	uint32_t gamesPerEvaluation = 100;
	float replaceRate = 0.55f;

	OptimizerConfig optimizerConfig;
};

class MCTSPlayer : public Player
{
public:
	static const int MAX_ACTION_COUNT = 10;

	MCTSPlayer(MCTSModel* model, int player, const Game& game, const MCTSConfig& config) : m_model(model), m_player(player), m_config(config)
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

		m_replayBuffer.turns.push_back({ game.getState(m_player), policy, Tensor({1,1,1}) });

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
		if (game->isFinished())
		{
			return game->getReward(m_player);
		}

		if (node->isLeaf)
		{
			float leafV = expand(game, node);
			if (game->getCurrentPlayer() != m_player)
				leafV = -leafV;
			return leafV;
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

	float expand(const Game* game, Node* node)
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


	// TODO: redo with pool allocator
	Node* createNode() { return new Node(); };
	void destroyNode(Node* node)
	{
		for (int i = 0; i < MAX_ACTION_COUNT; i++)
			if (node->links[i].child)
				destroyNode(node->links[i].child);
		delete node;
	}

	unique_ptr<Game> m_game;
	MCTSModel* m_model;
	int m_player;
	uint32_t m_actionCount;
	Node* m_currentNode;
	MCTSConfig m_config;
	uint32_t m_replayBufferStartIndex;
	ReplayBuffer m_replayBuffer;
};


static int playNGames(Game& game, Player* player0, Player* player1, uint32_t gameCount = 1)
{
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

	return player0Wins;
}

void checkCreateDir(fs::path dirPath)
{
	if (!fs::exists(dirPath))
		fs::create_directory(dirPath);
}

static bool loadBestModel(MCTSModel* model, const GlobalConfig& config)
{
	fs::path p = config.modelOutputPath;
	p /= config.bestModelFilename;
	if (!fs::exists(p))
		return false;
	model->model->load(p);
	return true;
}

static void saveAsBestModel(MCTSModel* model, const GlobalConfig& config)
{
	fs::path p = config.modelOutputPath;
	checkCreateDir(p);
	p /= config.bestModelFilename;
	model->model->save(p);
}

static void saveModel(MCTSModel* model, const GlobalConfig& config)
{
	fs::path p = config.modelOutputPath;
	checkCreateDir(p);
	p /= dateTimeNow();
	p += ".model";
	model->model->save(p);
}

static set<string, greater<string>> getDateDescendingSortedFiles(const fs::path p)
{
	set<string, greater<string>> s;
	for (auto& p : fs::directory_iterator(p))
		s.insert(p.path().string());
	return s;
}

class Worker
{
public:
	Worker(const string& name, GlobalConfig& config) : m_config(config)
	{
		fs::path p = config.logPath;
		checkCreateDir(p);
		p /= name;
		p += ".log";
		m_ofs.open(p, ifstream::app);
	}

	void start()
	{
		log("\n[Starting Session]");
		m_thread = std::thread(&Worker::run, this);
	}

protected:
	virtual void run() = 0;

	template <typename T, typename... Ts>
	void log(T head, Ts... tail)
	{
		m_ofs << dateTimeNow() << ":\t";
		logRecursive(head, tail...);
	}

	GlobalConfig m_config;

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

	std::ofstream m_ofs;
	std::thread m_thread;
};

class SelfPlayWorker : public Worker
{
public:
	SelfPlayWorker(GlobalConfig& config) : Worker("self_play", config) {}

	void run() override
	{
		// model
		auto model = m_config.buildModelFn(m_config.game);
		auto game = m_config.game->clone();

		while (1)
		{
			log("starting iteration");
			if (loadBestModel(model.get(), m_config))
				log("best model loaded");

			MCTSPlayer player0(model.get(), 0, *game, m_config.mctsConfig);
			MCTSPlayer player1(model.get(), 1, *game, m_config.mctsConfig);
			playNGames(*game, &player0, &player1, m_config.gamesPerReplayFile);

			checkCreateDir(m_config.replayOutputPath);
			fs::path filename = fs::path(m_config.replayOutputPath) / dateTimeNow();
			filename += ".bin";
			{
				ofstream ofs(filename, ifstream::out | ios::binary);
				player0.saveAndClearReplayBuffer(ofs);
				player1.saveAndClearReplayBuffer(ofs);

				log("saved replay: ", filename);
			}

			std::this_thread::sleep_for(10ms);
		}
	}
};

class OptimizeWorker : public Worker
{
public:
	OptimizeWorker(GlobalConfig& config) : Worker("optimizer", config) {}

	void run() override
	{
		// build model template
		auto model = m_config.buildModelFn(m_config.game);

		// init trainer
		AdamTrainer t;
		t.lr = 0.001f;
		t.batchSize = m_config.optimizerConfig.batchSize;
		t.init(model->model.get());

		while (1)
		{
			log("starting iteration");

			// load best model
			if (!loadBestModel(model.get(), m_config))
			{
				log("no best model yet, starting with random");
			}

			// load replay buffer (load most recent files till buffer is full, remove others)
			ReplayBuffer buffer;

			while (1)
			{
				auto sortedFiles = getDateDescendingSortedFiles(m_config.replayOutputPath);
				for (auto& file : sortedFiles)
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
					log("added replays from file", file);
				}

				if (!buffer.turns.empty())
					break;

				std::this_thread::sleep_for(1s); // wait and iterate till the first file
			}

			// optimize
			log("starting optimization");
			const uint32_t iterations = m_config.optimizerConfig.iterationCount;
			for (uint32_t i = 0; i < iterations; i++)
			{
				if (i % 100 == 0)
				{
					log("optimization loop ", i , " of ", iterations);
				}
				
				int sample = randUniform(0, (int)buffer.turns.size() - 1);
				t.train({ &buffer.turns[sample].state }, { &buffer.turns[sample].policy, &buffer.turns[sample].reward });
			}

			// save model
			fs::path p = m_config.modelOutputPath;
			checkCreateDir(p);
			p /= dateTimeNow();
			p += ".model";
			model->model->save(p);
			log("model saved: ", p);

			std::this_thread::sleep_for(100ms);
		}
	}
};

class EvaluateWorker : public Worker
{
public:
	EvaluateWorker(GlobalConfig& config) : Worker("evaluate", config) {}

	void run() override
	{
		auto game = m_config.game->clone();

		while (1)
		{
			log("starting iteration");

			// build and load best model
			auto bestModel = m_config.buildModelFn(m_config.game);
			if (!loadBestModel(bestModel.get(), m_config))
			{
				log("no best model yet, evaluating against random model");
			}

			auto candidateModel = m_config.buildModelFn(m_config.game);
			auto sortedFiles = getDateDescendingSortedFiles(m_config.modelOutputPath);
			for (auto& file : sortedFiles)
			{
				if((fs::path(file)).filename() == m_config.bestModelFilename)
					continue;
				candidateModel->model->load(file);
				log("evaluating:", file);

				MCTSPlayer player0(candidateModel.get(), 0, *game, m_config.mctsConfig);
				MCTSPlayer player1(bestModel.get(), 1, *game, m_config.mctsConfig);
				int player0Wins = playNGames(*game, &player0, &player1, m_config.gamesPerEvaluation);

				float winRate = player0Wins / static_cast<float>(m_config.gamesPerEvaluation);
				log("winrate: ", winRate);
				if (winRate >= m_config.replaceRate)
				{
					log("New best model: ", file, " win rate:", winRate);
					saveAsBestModel(candidateModel.get(), m_config);
				}

				fs::remove(file);
				break;
			}
			
			if (sortedFiles.size() <= 1)
			{
				log("no files to evaluate, waiting");
				std::this_thread::sleep_for(5s);
			}
		}
	}
};

static unique_ptr<MCTSModel> buildModel1(const Game* game)
{
	auto mctsModel = make_unique<MCTSModel>();

	// model
	Dim inputDim = game->getStateDim();
	auto input = make_shared<Input>(inputDim);
	auto x = (*make_shared<Conv>(16, 3, 3, 2, 1))(input);
	x = (*make_shared<Relu>())(x);
	x = (*make_shared<Conv>(16, 3, 3, 2, 1))(x);
	x = (*make_shared<Relu>())(x);
	auto split = (*make_shared<Dense>(32))(x);
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

	Connect4 game;

	GlobalConfig config;
	config.buildModelFn = buildModel1;
	config.game = &game;

	{
		SelfPlayWorker selfPlay(config);
		selfPlay.start();
		OptimizeWorker optimizer(config);
		optimizer.start();
		EvaluateWorker evaluate(config);
		evaluate.start();

		getchar();
	}

	return 0;
}