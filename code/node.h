#ifndef NODE_H
#define NODE_H

#include "beliefstate.h"
#include "utils.h"
#include <iostream>

class HISTORY;
class SIMULATOR;
class QNODE;
class VNODE;

//-----------------------------------------------------------------------------
// Efficient computation of value from alpha vectors
// Only used for explicit POMDPs
struct ALPHA
{
	std::vector<double> AlphaSum;
	double MaxValue;
};

//-----------------------------------------------------------------------------

template<class COUNT>
class VALUE
{
public:

	void Set(double count, double value)
	{
		Count = count;
		Total = value * count;
		SquaredTotal = value*value*count;
	}

	void Add(double totalReward)
	{
		Count += 1.0;
		Total += totalReward;
		SquaredTotal += totalReward*totalReward;
	}

	void Add(double totalReward, COUNT weight)
	{
		Count += weight;
		Total += totalReward * weight;
	}

	double GetValue() const
	{
		return Count == 0 ? Total : Total / Count;
	}

	COUNT GetCount() const
	{
		return Count;
	}

	double GetSquaredValue() const
	{
		return SquaredTotal;
	}

private:

	COUNT Count;
	double Total;
	double SquaredTotal;
};

//-----------------------------------------------------------------------------

class QNODE
{
public:

	VALUE<int> Value;
	VALUE<double> AMAF;
	double LowerBound;
	double UpperBound;
	int VisitCount;
	void Initialise();
	double Weight() const;
	double Gap() const { return UpperBound - LowerBound; }
	VNODE*& Child(int c) { return Children[c]; }
	VNODE* Child(int c) const { return Children[c]; }
	ALPHA& Alpha() { return AlphaData; }
	const ALPHA& Alpha() const { return AlphaData; }
	void IncrementChildrenCount(const int observation)
	{
		if(!initialized)
		{
			Initialise();
		}
		if(VisitCounts[observation] == 0)
		{
			ChildrenCount++;
		}
		VisitCounts[observation]++;
	}
	const int GetVisitCount(const int c) const { return VisitCounts[c]; }
	const int GetChildrenCount() const { return ChildrenCount; }
	void DisplayValue(HISTORY& history, int maxDepth, std::ostream& ostr) const;
	void DisplayPolicy(HISTORY& history, int maxDepth, std::ostream& ostr) const;
	bool IsLeaf() const { return Children.empty(); }
	static int NumChildren;
private:
	bool initialized = false;
	std::vector<int> VisitCounts;
	std::vector<VNODE*> Children;
	int ChildrenCount;
	ALPHA AlphaData;
	friend class VNODE;
};

//-----------------------------------------------------------------------------

class VNODE : public MEMORY_OBJECT
{
public:
	VALUE<int> Value;
	double LowerBound;
	double UpperBound;
	int VisitCount;
	void Initialise();
	static VNODE* Create();
	static void Free(VNODE* vnode, const SIMULATOR& simulator);
	static void FreeAll();
	double Weight() const { return BeliefState.GetNumScenarios()/500.0; }
	double Gap() const { return UpperBound - LowerBound; }
	QNODE& Child(int c) { return Children[c]; }
	const QNODE& Child(int c) const { return Children[c]; }
	BELIEF_STATE& Beliefs() { return BeliefState; }
	const BELIEF_STATE& Beliefs() const { return BeliefState; }
	bool IsLeaf() const { return Children.empty(); }
	void setBeliefs(BELIEF_STATE& newBelief)
	{
		BeliefState = newBelief;
	}

	void SetChildren(int count, double value);

	void DisplayValue(HISTORY& history, int maxDepth, std::ostream& ostr) const;
	void DisplayPolicy(HISTORY& history, int maxDepth, std::ostream& ostr) const;

	static int NumChildren;
private:
	std::vector<QNODE> Children;
	BELIEF_STATE BeliefState;
	static MEMORY_POOL<VNODE> VNodePool;
};

#endif // NODE_H
