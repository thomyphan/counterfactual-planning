#pragma once
#include "planner.h"
#include "causal_ucb.h"

using namespace std;

/**
 * CORAL node using MABUCs for tree traversal. 
 */
class CounterfactualPOOLTSNode : public POOLTSNode
{
public:
    CounterfactualPOOLTSNode(const SIMULATOR& simulator, const MCTS::PARAMS& params) : POOLTSNode(simulator, params)
    {
		this->counterfactualBandit = new MABUC(numberOfActions, 1, 1, params.BanditBetaPrior, int(params.NumSimulations*params.IntuitionLearningRatio));
    }

    ~CounterfactualPOOLTSNode()
    {
		delete counterfactualBandit;
    }

	virtual MABUC* getCounterfactualBandit()
    {
        return counterfactualBandit;
    }

    virtual int SelectAction()
    {
        const int action = this->counterfactualBandit->play();
        return action;
    }

    virtual void Update(const double reward)
    {
        counterfactualBandit->update(reward);
    }

    virtual void reset()
    {
    	POOLTSNode::reset();
        counterfactualBandit->reset();
    }

	POOLTSNode* getNext(const int action, std::list<POOLTSNode*>& pool)
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
				child = new CounterfactualPOOLTSNode(Simulator, Params);
			}
			children[action] = child;
		}
		return child;
    }

private:
    MABUC* counterfactualBandit;
};

/**
 * Counterfactual Open-loop Reasoning with Ad hoc Learning (CORAL)
 * as MABUC-based open-loop MCTS approach. 
 */
class CORAL : public POOLTS
{
public:
    CORAL(const SIMULATOR& simulator, const PARAMS& params) : POOLTS(simulator, params)
    {
    	this->rootNode = new CounterfactualPOOLTSNode(simulator, params);
    }
    virtual ~CORAL()
    {}
    virtual double Simulate(STATE& state, POOLTSNode* node, int t);
};

/**
 * COURAGE node using MABUCs for tree traversal. 
 */
class COURAGENode : public POOLTSNode
{
public:
    COURAGENode(const SIMULATOR& simulator, const MCTS::PARAMS& params) : POOLTSNode(simulator, params)
    {
		this->bandit = new CausalUCB(numberOfActions, 1, simulator.GetRewardRange(), int(params.NumSimulations*params.IntuitionLearningRatio));
    }

    ~COURAGENode()
    {
    }

	POOLTSNode* getNext(const int action, std::list<POOLTSNode*>& pool)
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
				child = new COURAGENode(Simulator, Params);
			}
			children[action] = child;
		}
		return child;
    }
};

/**
 * Causal and Open-loop UCB Reasoning for Advanced Gain Estimates (COURAGE)
 * as MABUC-based open-loop MCTS approach. 
 */
class COURAGE : public POOLTS
{
public:
    COURAGE(const SIMULATOR& simulator, const PARAMS& params) : POOLTS(simulator, params)
    {
    	this->rootNode = new COURAGENode(simulator, params);
    }
    virtual ~COURAGE()
    {}
    virtual double Simulate(STATE& state, POOLTSNode* node, int t)
    {
        std::vector<int> legal;
        Simulator.GenerateLegal(state, GetHistory(), legal, GetStatus());
        std::vector<int> heuristic;
        if(Params.HumanKnowledge)
        {
            Simulator.GeneratePreferred(state, GetHistory(), heuristic, GetStatus());
        }
        else
        {
            heuristic = legal;
        }

        if(heuristic.empty())
        {
            heuristic = legal;
        }
        int action;
        Bandit* bandit = node->getBandit();
        if(bandit->isWarmingUp())
        {
            action = bandit->sampleFrom(legal);
        }
        else
        {
            action = bandit->sampleFrom(legal);
        }
        PeakTreeDepth = TreeDepth;
        if (t >= Params.MaxDepth)
        {
            return 0;
        }
        bool isLeaf = node->IsLeaf();
        if(isLeaf)
        {
            node->Expand();
            IncrementNodeCountBy(Simulator.GetNumActions());
        }
        int observation;
        double immediateReward, delayedReward = 0;
        bool terminal = Simulator.Step(state, action, observation, immediateReward);
        if(t == 0)
        {
            VNODE*& vnode = Root->Child(action).Child(observation);
            if (!vnode && !terminal) {
                vnode = ExpandNode(&state);
                AddSample(vnode, state);
        }
        }
        History.Add(action, observation);
        if(terminal)
        {
            node->Update(immediateReward);
            return immediateReward;
        }
        assert(observation >= 0 && observation < Simulator.GetNumObservations());

        TreeDepth++;
        delayedReward = isLeaf? Rollout(state) : Simulate(state, node->getNext(action, pool), t+1);
        TreeDepth--;

        double totalReward = immediateReward + Simulator.GetDiscount() * delayedReward;
        node->Update(totalReward);
        return totalReward;
    }
};
