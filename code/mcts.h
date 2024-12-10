#ifndef MCTS_H
#define MCTS_H

#include "simulator.h"
#include "node.h"
#include "statistic.h"
#include <boost/random.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <random>
#include <chrono>

class MCTS
{
public:
	struct PARAMS
	{
		PARAMS();

		int Verbose;
		int MaxDepth;
		int NumSimulations;
		int NumStartStates;
		bool UseTransforms;
		int NumTransforms;
		int MaxAttempts;
		int ExpandCount;
		int EnsembleSize;
        int BanditArmCapacity;
        double BanditConvergenceEpsilon;
		int BanditBetaPrior;
		double ExplorationConstant;
		bool UseRave;
		double RaveDiscount;
		double RaveConstant;
		bool DisableTree;
		bool PreferredActions;
		bool UseThompsonSampling;
		int SelectionKnowledge;
		int HumanKnowledge;
		double kObservations;
		double alphaObservations;
		double IntuitionLearningRatio;
	};

	MCTS(const SIMULATOR& simulator, const PARAMS& params);
	virtual ~MCTS();

	virtual int SelectAction();
	virtual int SampleAction(VNODE* vnode, STATE& state, bool ucb) const;
	virtual bool Update(int action, int observation, double reward);

	void UCTSearch();
	void RolloutSearch();

	double Rollout(STATE& state);

	const BELIEF_STATE& BeliefState() const { return Root->Beliefs(); }
	const HISTORY& GetHistory() const { return History; }
	const SIMULATOR::STATUS& GetStatus() const { return Status; }
	void ClearStatistics();
	void DisplayStatistics(std::ostream& ostr) const;
	void DisplayValue(int depth, std::ostream& ostr) const;
	void DisplayPolicy(int depth, std::ostream& ostr) const;

	static void UnitTest();
	static void InitFastUCB(double exploration);

	int GreedyUCB(VNODE* vnode, bool ucb, STATE& state) const;
	int ThompsonSamplingAction(VNODE* vnode, STATE& state) const;
	int SelectRandom() const;
	virtual int GetNodeCount() const
	{
		return nodeCount;
	}
	virtual void IncrementNodeCountBy(const int addition)
	{
		nodeCount += addition;
	}

	virtual void IncrementNodeCountStatistics()
	{
		nodeCountStatistics.Add(nodeCount);
	}

	virtual double GetMeanNodeCount()
	{
		nodeCountStatistics.Add(nodeCount);
		return nodeCountStatistics.GetMean();
	}
	virtual double SimulateV(STATE& state, VNODE* vnode);
	virtual double SimulateQ(STATE& state, QNODE& qnode, int action);
	void AddRave(VNODE* vnode, double totalReward);
	virtual VNODE* ExpandNode(const STATE* state);
	virtual void AddSample(VNODE* node, const STATE& state);
	void AddTransforms(VNODE* root, BELIEF_STATE& beliefs);
	STATE* CreateTransform() const;
	void Resample(BELIEF_STATE& beliefs);

	// Fast lookup table for UCB
	static const int UCB_N = 10000, UCB_n = 100;
	static double UCB[UCB_N][UCB_n];
	static bool InitialisedFastUCB;

	double FastUCB(int N, int n, double logN) const;
	const SIMULATOR& Simulator;
	int TreeDepth, PeakTreeDepth;
	PARAMS Params;
	VNODE* Root;
	HISTORY History;
	SIMULATOR::STATUS Status;
	STATISTIC StatTreeDepth;
	STATISTIC StatRolloutDepth;
	STATISTIC StatTotalReward;
protected:
	STATISTIC nodeCountStatistics;
private:
	static void UnitTestGreedy();
	static void UnitTestUCB();
	static void UnitTestRollout();
	static void UnitTestSearch(int depth);
	double mu0 = 0;
	double lambda0 = 0.01;
	double alpha0 = 1;
	double beta0 = 1000;
	int nodeCount;
};

#endif // MCTS_H
