#pragma once
#include "mcts.h"
#include "bandit.h"
#include "mabuc.h"
#include <fstream>
#include <list>
#include <algorithm>
#include <iostream>

using namespace std;

class POSTS : public MCTS
{
public:
	POSTS(const SIMULATOR& simulator, const PARAMS& params) : MCTS(simulator, params), currentIndex(0), stackSize(params.MaxDepth/5)
	{
		for (int t = 0; t < Params.MaxDepth; t++)
		{
			bandits.push_back(new ThompsonSampling(Simulator.GetNumActions(), 0, 1, params.BanditBetaPrior));
		}
	}
	virtual ~POSTS()
	{
		for (int t = 0; t < Params.MaxDepth; t++)
		{
			if(bandits[t] != NULL)
			{
				delete bandits[t];
			}
		}
	}
	void reset()
	{
		for (int t = 0; t < Params.MaxDepth; t++)
		{
			bandits[t]->reset();
		}
	}
	virtual double GetMeanNodeCount()
	{
		return Params.MaxDepth;
	}
	virtual int SampleAction(const int t, const int banditIndex, STATE& state, std::vector<int>& legalActions);
	virtual int SelectAction();
	double Rollout(STATE& state, std::vector<int>& legalActions, const int t, const int i);
	void Rollout();
protected:
	int currentIndex;
	const int stackSize;
	std::vector<ThompsonSampling*> bandits;
};

class POOLTSNode
{
public:
    POOLTSNode(const SIMULATOR& simulator, const MCTS::PARAMS& params) : Simulator(simulator), Params(params), numberOfActions(simulator.GetNumActions())
    {
		this->bandit = new ThompsonSampling(numberOfActions, 1, 1, params.BanditBetaPrior);
    }

    virtual ~POOLTSNode()
    {
		delete bandit;
        int children_size = children.size();
        for(int index = 0; index < children_size; index++)
        {
            if(this->children[index] != NULL)
	    {
                delete this->children[index];
	    }
        }
        this->children.clear();
    }

    virtual int SelectAction()
    {
        return this->bandit->play();
    }

    void Expand()
	{
		if(this->children.empty())
		{
			for(int index = 0; index < numberOfActions; index++)
			{
				this->children.push_back(NULL);
			}
		}
        this->isLeafNode = false;
    }

    Bandit* getBandit()
    {
        return bandit;
    }

	virtual MABUC* getCounterfactualBandit()
    {
        return NULL;
    }

    virtual void Update(const double reward)
    {
        bandit->update(reward);
    }

    virtual POOLTSNode* getNext(const int action, std::list<POOLTSNode*>& pool)
    {
        POOLTSNode* child = children[action];
		if(child == NULL)
		{
			if(!pool.empty())
			{
				child = pool.front();
			pool.pop_front();
			}
			else
			{
				child = new POOLTSNode(Simulator, Params);
			}
			children[action] = child;
		}
		return child;
    }
    void resetNext(const int action)
    {
    	this->children[action] = NULL;
    }
    const bool IsLeaf()
    {
        return this->isLeafNode;
    }
    virtual void reset()
    {
    	this->isLeafNode = true;
    	this->bandit->reset();
    }
    void saveToPool(std::list<POOLTSNode*>& pool)
    {
		int numberOfChildren = this->children.size();
			for(int index = 0; index < numberOfChildren; index++)
		{
				if(this->children[index] != NULL)
			{
					this->children[index]->saveToPool(pool);
			}
		}
		this->children.clear();
		this->reset();
		pool.push_back(this);
    }
protected:
    Bandit* bandit;
    std::vector<POOLTSNode*> children;
    bool isLeafNode = true;
    const int numberOfActions;
    const SIMULATOR& Simulator;
    const MCTS::PARAMS& Params;
};

/**
 * Standard open-loop MCTS. Can traverse the search tree using human knowledge in form of preferred actions. 
 */
class POOLTS : public MCTS
{
public:
    POOLTS(const SIMULATOR& simulator, const PARAMS& params) : MCTS(simulator, params)
    {
    	this->rootNode = new POOLTSNode(simulator, params);
    }
    virtual ~POOLTS()
    {
        delete rootNode;
		int pool_size = pool.size();
		for(int index = 0; index < pool_size; index++)
		{
			POOLTSNode* next = pool.front();
			delete next;
			pool.pop_front();
		}
		pool.clear();
    }
    virtual int SelectAction()
    {
		TreeSearch();
		int action = rootNode->SelectAction();
		rootNode->saveToPool(pool);
		rootNode = pool.front();
		pool.pop_front();
		IncrementNodeCountStatistics();
		rootNode->reset();
		return action;
    }
	virtual void IncrementNodeCountBy(const int addition)
	{
		openLoopNodeCount += addition;
	}
	virtual void IncrementNodeCountStatistics()
	{
		nodeCountStatistics.Add(openLoopNodeCount);
		openLoopNodeCount = 1;
	}
	virtual int GetNodeCount() const
	{
		return openLoopNodeCount;
	}
	virtual double GetMeanNodeCount()
	{
		return nodeCountStatistics.GetMean();
	}
    virtual void TreeSearch();
    virtual double Simulate(STATE& state, POOLTSNode* node, int t);
protected:
    POOLTSNode* rootNode;
    std::list<POOLTSNode*> pool;
	int openLoopNodeCount;
};

class SYMBOL : public MCTS
{
public:
	SYMBOL(const SIMULATOR& simulator, const PARAMS& params) : MCTS(simulator, params), currentIndex(0), banditConvergenceEpsilon(params.BanditConvergenceEpsilon), maxNumberOfBandits(0)
	{
		for (int t = 0; t < Params.MaxDepth; t++)
		{
			bandits.push_back(new ThompsonSampling(Simulator.GetNumActions(), Params.BanditArmCapacity, 1, params.BanditBetaPrior));
            rewards.push_back(0.0);
		}
	}
	virtual ~SYMBOL()
	{
		for (int t = 0; t < Params.MaxDepth; t++)
		{
			delete bandits[t];
		}
	}
	void reset()
	{
		for (int t = 0; t < Params.MaxDepth; t++)
		{
			bandits[t]->reset();
            rewards[t] = 0.0;
		}
        maxNumberOfBandits = 0;
	}
	virtual int SelectAction();
	double Rollout(STATE& state, std::vector<int>& legalActions, const int t, const int i);
    const int getMaxNumberOfBandits()
    {
        return maxNumberOfBandits;
    }
	void Rollout();
private:
	int currentIndex;
    double banditConvergenceEpsilon;
    int maxNumberOfBandits;
	std::vector<ThompsonSampling*> bandits;
    std::vector<double> rewards;
};
