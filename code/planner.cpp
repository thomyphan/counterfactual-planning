#include "planner.h"

int POSTS::SelectAction()
{
	int banditIndex = currentIndex%Params.MaxDepth;
	reset();
	Rollout();
	int numberOfActions = Simulator.GetNumActions();
	return GreedyUCB(Root, false, *Simulator.CreateStartState());
}

void POSTS::Rollout()
{
	std::vector<double> totals(Simulator.GetNumActions(), 0.0);
	int historyDepth = History.Size();
	assert(BeliefState().GetNumSamples() > 0);
	for (int i = 0; i < Params.NumSimulations; i++)
	{
		std::vector<int> legalActions;
		STATE* state = Root->Beliefs().CreateSample(Simulator);
		int banditIndex = currentIndex%Params.MaxDepth;
		int action = SampleAction(0, banditIndex, *state, legalActions);
		Simulator.Validate(*state);

		int observation;
		double immediateReward, delayedReward, totalReward;
		bool terminal = Simulator.Step(*state, action, observation, immediateReward);

		VNODE*& vnode = Root->Child(action).Child(observation);
		if (!vnode && !terminal)
		{
			vnode = ExpandNode(state);
			AddSample(vnode, *state);
		}
		History.Add(action, observation);
		delayedReward = Rollout(*state, legalActions, 1,i);
		totalReward = immediateReward + Simulator.GetDiscount() * delayedReward;
		Root->Child(action).Value.Add(totalReward);
		bandits[banditIndex]->update(totalReward);
		Simulator.FreeState(state);
		History.Truncate(historyDepth);
	}
}

int POSTS::SampleAction(const int t, const int banditIndex, STATE& state, std::vector<int>& legalActions)
{
	legalActions.clear();
	if(Params.SelectionKnowledge == SIMULATOR::KNOWLEDGE::SMART && t < stackSize)
	{
		Simulator.GeneratePreferred(state, GetHistory(), legalActions, GetStatus());
	}
		
	if(Params.SelectionKnowledge != SIMULATOR::KNOWLEDGE::SMART || legalActions.empty())
	{
		Simulator.GenerateLegal(state, GetHistory(), legalActions, GetStatus());
	}
	if(t >= stackSize)
	{
		return legalActions[randomInt(legalActions.size())];
	}
	return bandits[banditIndex]->sampleFrom(legalActions);
}

double POSTS::Rollout(
	STATE& state,
	std::vector<int>& legalActions,
	const int t,
	const int i)
{
	if (t >= Params.MaxDepth)
	{
		return 0;
	}
	int numberOfActions = Simulator.GetNumActions();
	bool terminal = false;
	int observation;
	double immediateReward, delayedReward;
	int banditIndex = (currentIndex + t)%Params.MaxDepth;
	if (!terminal) {
		int action = SampleAction(t, banditIndex, state, legalActions);
		terminal = Simulator.Step(state, action, observation, immediateReward);
		History.Add(action, observation);
	}
	if (terminal)
	{
		if(t < stackSize)
		{
	    	bandits[banditIndex]->update(immediateReward);
		}
		return immediateReward;
	}
	double successorReturn = Rollout(state, legalActions, t+1,i);
	double discount = Simulator.GetDiscount();
	double returnValue = immediateReward + discount*successorReturn;
	if(t < stackSize)
	{
		bandits[banditIndex]->update(returnValue);
	}
	return returnValue;
}

void POOLTS::TreeSearch()
{
	int historyDepth = History.Size();
	for (int n = 0; n < Params.NumSimulations; n++)
	{
		STATE* state = Root->Beliefs().CreateSample(Simulator);
		Simulator.Validate(*state);
		Status.Phase = SIMULATOR::STATUS::TREE;
		TreeDepth = 0;
		PeakTreeDepth = 0;
		double totalReward = Simulate(*state, rootNode, 0);
		StatTotalReward.Add(totalReward);
		StatTreeDepth.Add(PeakTreeDepth);
		Simulator.FreeState(state);
		History.Truncate(historyDepth);
	}
}

double POOLTS::Simulate(STATE& state, POOLTSNode* node, int t)
{
	std::vector<int> legal;
	if(Params.SelectionKnowledge == SIMULATOR::KNOWLEDGE::SMART && t > 0)
	{
		Simulator.GeneratePreferred(state, GetHistory(), legal, GetStatus());
	}
	
	if(Params.SelectionKnowledge != SIMULATOR::KNOWLEDGE::SMART || legal.empty()){
		Simulator.GenerateLegal(state, GetHistory(), legal, GetStatus());
	}
    int action = node->getBandit()->sampleFrom(legal);
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

int SYMBOL::SelectAction()
{
	reset();
	Rollout();
	return GreedyUCB(Root, false, *Simulator.CreateStartState());
}

void SYMBOL::Rollout()
{
	std::vector<double> totals(Simulator.GetNumActions(), 0.0);
	int historyDepth = History.Size();
	std::vector<int> legal;
	assert(BeliefState().GetNumSamples() > 0);
	for (int i = 0; i < Params.NumSimulations; i++)
	{
		STATE* state = Root->Beliefs().CreateSample(Simulator);
        STATE* firstState = state;
		Simulator.GenerateActionSpace(*state, GetHistory(), legal, GetStatus(), Params.PreferredActions);
		
		int action = bandits[0]->sampleFrom(legal);
        int firstAction = action;
		Simulator.Validate(*state);

		int observation;
		double immediateReward;
		bool terminal = Simulator.Step(*state, action, observation, immediateReward);

		VNODE*& vnode = Root->Child(action).Child(observation);
		if (!vnode && !terminal)
		{
			vnode = ExpandNode(state);
			AddSample(vnode, *state);
		}
		History.Add(action, observation);
        rewards[0] = immediateReward;
        int stepCount = 1;
        for(int t = 1; t < Params.MaxDepth; t++)
        {
            if(!terminal)
            {
                Simulator.GenerateActionSpace(*state, GetHistory(), legal, GetStatus(), Params.PreferredActions);
                int action = bandits[t]->sampleFrom(legal);
                terminal = Simulator.Step(*state, action, observation, immediateReward);
                History.Add(action, observation);
                rewards[stepCount] = immediateReward;
                stepCount += 1;
            }
        }
        double returnValue = 0;
        for(int t = stepCount - 1; t >= 0; t--)
        {
            returnValue = rewards[t] + Simulator.GetDiscount()*returnValue;
            rewards[t] = returnValue;
        }
		Root->Child(firstAction).Value.Add(rewards[0]);
        bandits[0]->update(rewards[0]);
        bool predecessorConverged = bandits[0]->hasConverged(banditConvergenceEpsilon);
        int numberOfBandits = 1;
        for(int t = 1; t < stepCount; t++)
        {
            if(predecessorConverged)
            {
                bandits[t]->update(rewards[t]);
                numberOfBandits += 1;
                predecessorConverged = bandits[t]->hasConverged(banditConvergenceEpsilon);
            }
        }
		maxNumberOfBandits = std::max(maxNumberOfBandits, numberOfBandits);
		Simulator.FreeState(firstState);
		History.Truncate(historyDepth);
	}
}