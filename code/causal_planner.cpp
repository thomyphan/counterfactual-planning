#include "causal_planner.h"

// Counterfactual simulation, according to Algorithm 2 in our paper
double CORAL::Simulate(STATE& state, POOLTSNode* node, int t)
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
    int action = node->getCounterfactualBandit()->sampleCounterfactualFrom(heuristic, legal);
    IncrementNodeCountBy(node->getCounterfactualBandit()->GetNewBanditCount());
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