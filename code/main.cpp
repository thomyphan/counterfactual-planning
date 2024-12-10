#include "battleship.h"
#include "mcts.h"
#include "network.h"
#include "pocman.h"
#include "rocksample.h"
#include "tag.h"
#include "experiment.h"
#include <string>
#include <boost/program_options.hpp>

using namespace std;
using namespace boost::program_options;

int main(int argc, char* argv[])
{
	MCTS::PARAMS searchParams;
	EXPERIMENT::PARAMS expParams;
	SIMULATOR::KNOWLEDGE knowledge;
	knowledge.RolloutLevel = SIMULATOR::KNOWLEDGE::LEGAL;
	string problem, policy, horizonString, algorithmName, humanKnowledge;
	string outputFilePrefix;
	string outputfile;
	string nodecountfile;
    string banditArmCapacity, banditBetaPriorString, banditConvergenceEpsilonString, learningRatio;
	int size, number, treeknowledge = 1, rolloutknowledge = 1, smarttreecount = 10;
	learningRatio = "0.5";
	problem = argv[1];
	humanKnowledge = "Legal";
	algorithmName = "MCTS";
	if(argc > 2)
	{
		algorithmName = argv[2];
		if(argc > 3)
		{
			if(string(argv[3]) == "True")
			{
				searchParams.SelectionKnowledge = SIMULATOR::KNOWLEDGE::SMART;
				searchParams.PreferredActions = true;
				searchParams.HumanKnowledge = true;
				humanKnowledge = "Smart";
			}
			else
			{
				searchParams.SelectionKnowledge = SIMULATOR::KNOWLEDGE::LEGAL;
				searchParams.PreferredActions = false;
				searchParams.HumanKnowledge = false;
				humanKnowledge = "Legal";
			}
			if(argc > 4)
			{
				learningRatio = argv[4];
				searchParams.IntuitionLearningRatio = stod(argv[4]);
			}
		}
	}
    SIMULATOR* real = 0;
	SIMULATOR* simulator = 0;
	if(problem == "battleship")
	{
		real = new BATTLESHIP(10, 10, 5);
		simulator = new BATTLESHIP(10, 10, 5);
	}
	else if(problem == "pocman")
	{
		real = new FULL_POCMAN();
		simulator = new FULL_POCMAN();
	}
	else if(problem == "network")
	{
		real = new NETWORK(size, number);
		simulator = new NETWORK(size, number);
	}
	else if(problem == "rocksample-11")
	{
        real = new ROCKSAMPLE(11, 11);
		simulator = new ROCKSAMPLE(11, 11);
	}
	else if(problem == "rocksample-15")
	{
        real = new ROCKSAMPLE(15, 15);
		simulator = new ROCKSAMPLE(15, 15);
	}
	else if(problem == "tag")
	{
		real = new TAG(1);
		simulator = new TAG(1);
	}
	else
	{
		cout << "Unknown problem" << endl;
		exit(1);
	}
	expParams = EXPERIMENT::PARAMS();
    searchParams.BanditArmCapacity = 8;
    searchParams.BanditBetaPrior = 1000;
	outputFilePrefix = problem + "_"+ algorithmName + "_" + humanKnowledge + "_" + learningRatio;
	if(!searchParams.HumanKnowledge)
	{
		outputFilePrefix += "_NoHeuristics";
	}
	outputfile = outputFilePrefix + ".csv";
	nodecountfile = outputFilePrefix + "_nodeCount.csv";
	cout << "outputfile: " << outputfile << endl;
	expParams.AlgorithmName = algorithmName;
    searchParams.MaxDepth = 100;
    searchParams.BanditConvergenceEpsilon = 1.0;
	simulator->SetKnowledge(knowledge);
	EXPERIMENT experiment(*real,*simulator, outputfile, nodecountfile, expParams, searchParams);
	experiment.DiscountedReturn();
	delete real;
	delete simulator;
	return 0;
}
