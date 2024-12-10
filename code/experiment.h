#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include "mcts.h"
#include "simulator.h"
#include "statistic.h"
#include <fstream>
#include <string>
#include "planner.h"
#include "causal_planner.h"

using namespace std;

//----------------------------------------------------------------------------

struct RESULTS
{
	void Clear();

	STATISTIC Time;
	STATISTIC Reward;
	STATISTIC DiscountedReturn;
	STATISTIC UndiscountedReturn;
	STATISTIC NodeCount;
};

inline void RESULTS::Clear()
{
	Time.Clear();
	Reward.Clear();
	DiscountedReturn.Clear();
	UndiscountedReturn.Clear();
}

//----------------------------------------------------------------------------

class EXPERIMENT
{
public:

	struct PARAMS
	{
		PARAMS();

		int NumRuns;
		int NumSteps;
		int SimSteps;
		double TimeOut;
		int MinDoubles, MaxDoubles;
		int TransformDoubles;
		int TransformAttempts;
		double Accuracy;
		int UndiscountedHorizon;
		bool AutoExploration;
		string AlgorithmName;
	};

	EXPERIMENT(const SIMULATOR& real, const SIMULATOR& simulator,
		const std::string& outputFile,
		const std::string& nodeCountFile,
		EXPERIMENT::PARAMS& expParams, MCTS::PARAMS& searchParams);

	void Run();
	void MultiRun();
	void DiscountedReturn();
	void AverageReward();

private:

	const SIMULATOR& Real;
	const SIMULATOR& Simulator;
	EXPERIMENT::PARAMS& ExpParams;
	MCTS::PARAMS& SearchParams;
	RESULTS Results;

	std::ofstream OutputFile;
	std::ofstream NodeCountFile;
};

//----------------------------------------------------------------------------

#endif // EXPERIMENT_H