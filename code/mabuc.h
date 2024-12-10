#ifndef MABUC_H
#define MABUC_H

#include "bandit.h"

// Counterfactual arm, according to Figure 3 in our paper.
class CounterfactualArm : public Arm
{
public:
	CounterfactualArm(const unsigned int numberOfArms,
		const unsigned int rewardBufferSize,
        const unsigned int updateDelay,
		const unsigned int beta0) : Arm(rewardBufferSize)
	{
		interventionalBandit = new ThompsonSampling(numberOfArms, rewardBufferSize, updateDelay, beta0);
	}

	virtual ~CounterfactualArm()
	{
		delete interventionalBandit;
	}

	int play()
	{
		return interventionalBandit->play();
	}

	int intervene(const std::vector<int>& legalArms)
	{
		return interventionalBandit->sampleFrom(legalArms);
	}

	void update(const double reward)
	{
		Arm::update(reward);
		interventionalBandit->update(reward);
	}

	virtual void reset()
	{
		Arm::reset();
		interventionalBandit->reset();
	}

	int interventionIndex() const
	{
		return interventionalBandit->currentPlayIndex();
	}

	int interventionCount(const int action) const
	{
		return interventionalBandit->getArm(action)->size();
	}

	double interventionMean(const int action) const
	{
		return interventionalBandit->getArm(action)->mean();
	}

	void setInterventionIndex(const int index)
	{
		interventionalBandit->setPlayIndex(index);
	}
private:
	ThompsonSampling* interventionalBandit;
};

// MABUC implementation, according to Figure 3 in our paper.
class MABUC : public ThompsonSampling 
{
public:
	MABUC(
		const unsigned int numberOfArms,
		const unsigned int rewardBufferSize,
        const unsigned int updateDelay,
		const unsigned int beta0,
		const unsigned int warmupPhase) : ThompsonSampling(numberOfArms, rewardBufferSize, updateDelay, beta0), warmupPhase(warmupPhase), newBanditCount(0), noHeuristic(false)
		{
			intuitionCounts.assign(numberOfArms, 0);
			interventionCounts.assign(numberOfArms, 0);
			newBanditCounts.assign(numberOfArms, 0);
			for(int index = 0; index < numberOfArms; index++)
			{
				Arm* oldArm = arms[index];
				CounterfactualArm* newArm = new CounterfactualArm(numberOfArms, rewardBufferSize, updateDelay, beta0);
				arms[index] = newArm;
				counterfactualArms.push_back(newArm);
				delete oldArm;
			}
		}

	virtual ~MABUC()
	{
		for(int index = 0; index < numberOfArms; index++)
		{
			arms[index] = NULL;
			delete counterfactualArms[index];
		}
	}
	virtual bool isWarmingUp()
	{
		return warmupPhase <= 0;
	}
	virtual int play()
	{
		const int intent = ThompsonSampling::play();
		if(warmupPhase > 0)
		{
			return intent;
		}
		return counterfactualArms[intent]->play();
	}

	virtual void update(const double reward)
	{
		int actionIndex = playIndex;
		if(warmupPhase <= 0 && !noHeuristic)
		{
			actionIndex = counterfactualArms[playIndex]->interventionIndex();
		}
		else
		{
			counterfactualArms[playIndex]->setInterventionIndex(playIndex);
		}
		interventionCounts[actionIndex] += 1;
		counterfactualArms[playIndex]->update(reward);
		means.clear();
		vars.clear();
		means.assign(numberOfArms, 0);
		vars.assign(numberOfArms, 0);
		if(playIndex == actionIndex || warmupPhase > 0 || noHeuristic)
		{
			intuitionCounts[playIndex] += 1;
			ThompsonSampling::update(reward);
		}
		noHeuristic = false;
		warmupPhase = max(0, warmupPhase - 1);
	}

	int sampleCounterfactualFrom(
		const std::vector<int>& heuristicArms,
		const std::vector<int>& legalArms)
	{
		static vector<double> intuitiveValues;
		intuitiveValues.clear();
		intuitiveArms.clear();
		intuitiveArms = heuristicArms;
		noHeuristic = heuristicArms.size() == legalArms.size();
		if(warmupPhase > 0 || noHeuristic)
		{
			playIndex = ThompsonSampling::sampleFrom(heuristicArms);
			return playIndex;
		}
		for(int intent : intuitiveArms)
		{
			intuitiveValues.push_back(counterfactualArms[intent]->mean());
		}
		playIndex = intuitiveArms[argmax(intuitiveValues)];
		const int oldCount = newBanditCounts[playIndex];
		if(oldCount == 0)
		{
			newBanditCounts[playIndex] += 1;
			newBanditCount += 1;
		}
		const int interventionalAction = counterfactualArms[playIndex]->intervene(legalArms);
		return interventionalAction;
	}

	int GetNewBanditCount()
	{
		const int newCount = newBanditCount;
		newBanditCount = 0;
		return newCount;
	}

	virtual void reset()
	{
		ThompsonSampling::reset();
		newBanditCounts.clear();
		newBanditCounts.assign(numberOfArms, 0);
		newBanditCount = 0;
	}
private:
	vector<int> intuitionCounts;
	vector<int> interventionCounts;
	vector<int> newBanditCounts;
	bool noHeuristic;
	int newBanditCount;
	int warmupPhase;
	vector<CounterfactualArm*> counterfactualArms;
	vector<int> intuitiveArms;
};

#endif // MABUC