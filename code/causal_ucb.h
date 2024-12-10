#ifndef CAUSALUCB_H
#define CAUSALUCB_H

#include "bandit.h"

class CausalUCB : public UCB1 
{
public:
	CausalUCB(
		const unsigned int numberOfArms,
		const unsigned int rewardBufferSize,
		double explorationConstant,
		const unsigned int warmupPhase) : UCB1(numberOfArms, rewardBufferSize, explorationConstant), warmupPhase(warmupPhase), newBanditCount(0), noHeuristic(false)
		{
			intentBandit = new UCB1(numberOfArms, rewardBufferSize, explorationConstant);
			lowerBounds.assign(numberOfArms, -explorationConstant);
			upperBounds.assign(numberOfArms, explorationConstant);
		}

	virtual ~CausalUCB()
	{
		delete intentBandit;
	}

	virtual int play()
	{
		const int intent = intentBandit->play();
		if(warmupPhase > 0)
		{
			return intent;
		}
		return UCB1::play();
	}

	virtual void update(const double reward)
	{
		if(warmupPhase > 0 || noHeuristic)
		{
			intentBandit->update(reward);
		}
		else
		{
			UCB1::update(reward);
		}
		noHeuristic = false;
		warmupPhase = max(0, warmupPhase - 1);
	}

	virtual int sampleArmFrom(const std::vector<int>& legalArms)
	{
		if(warmupPhase > 0 || noHeuristic)
		{
			return intentBandit->sampleFrom(legalArms);
		}
		upperConfidences.clear();
		const int numberOfArms = legalArms.size();
		int totalCount = 0;
		double maxLowerBound = -std::numeric_limits<double>::infinity();
		for(int index = 0; index < numberOfArms; index++)
		{
			Arm* arm = getArm(legalArms[index]);
			totalCount += arm->size();
			const double mean = arm->mean();
			const double std = arm->std();
			const double lowerBound = mean - 2*std;
			const double upperBound = mean + 2*std;
			if(lowerBound > -explorationConstant)
			{
				lowerBounds[index] = lowerBound;
			}
			if(lowerBound > maxLowerBound)
			{
				maxLowerBound = lowerBound;
			}
			if(upperBound < explorationConstant)
			{
				upperBounds[index] = upperBound;
			}
		}
		for (int index = 0; index < numberOfArms; index++)
		{
			Arm* arm = getArm(legalArms[index]);
			const double meanReward = arm->mean();
			const int numberOfRewards = arm->size();
			if (numberOfRewards == 0)
			{
				upperConfidences.push_back(std::numeric_limits<double>::infinity());
			}
			else if(upperBounds[index] < maxLowerBound)
			{
				upperConfidences.push_back(-std::numeric_limits<double>::infinity());
			}
			else
			{
				const double explorationTerm = sqrt(2 * log(totalCount) / numberOfArms);
				const double originalUCB1Term = meanReward + explorationConstant*explorationTerm;
				if(originalUCB1Term < upperBounds[index])
				{
					upperConfidences.push_back(originalUCB1Term);
				}
				else
				{
					upperConfidences.push_back(upperBounds[index]);
				}
			}
		}
		playIndex = legalArms[argmax(upperConfidences)];
		return playIndex;
	}

	int GetNewBanditCount()
	{
		const int newCount = newBanditCount;
		newBanditCount = 0;
		return newCount;
	}
	virtual bool isWarmingUp()
	{
		return warmupPhase <= 0;
	}

	virtual void reset()
	{
		UCB1::reset();
		lowerBounds.clear();
		lowerBounds.assign(numberOfArms, -explorationConstant);
		upperBounds.clear();
		upperBounds.assign(numberOfArms, explorationConstant);
		newBanditCount = 0;
	}
private:
	bool noHeuristic;
	int newBanditCount;
	int warmupPhase;
	vector<double> lowerBounds;
	vector<double> upperBounds;
	UCB1* intentBandit;
	vector<int> intuitiveArms;
};

#endif // MABUC