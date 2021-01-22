//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "SPN/Analysis/SLP/SLPTree.h"
#include "SPN/SPNOps.h"
#include "SPN/SPNOpTraits.h"

#include <iostream>
#include <algorithm>

using namespace mlir;
using namespace mlir::spn;
using namespace mlir::spn::slp;

SLPTree::SLPTree(Operation* root, size_t width, size_t maxLookAhead) : graphs{}, maxLookAhead{maxLookAhead} {
  assert(root);
  llvm::StringMap<std::vector<Operation*>> operationsByOpCode;
  std::set<Operation*> operations;
  for (auto& op : root->getBlock()->getOperations()) {
    operationsByOpCode[op.getName().getStringRef().str()].emplace_back(&op);
    operations.insert(&op);
    for (auto const& operand : op.getOperands()) {
      if (operations.find(operand.getDefiningOp()) != std::end(operations)) {
        std::cout << "it's not a tree!" << std::endl;
        return;
      }
    }
  }

  // TODO compute seeds in a proper fashion
  for (auto const& entry : operationsByOpCode) {
    SLPNode rootNode{entry.getValue()};
    buildGraph(entry.getValue(), rootNode);
  }

}

/*
void SLPTree::analyzeGraph(Operation* root) {
  traverseSubgraph(root);
}

void SLPTree::traverseSubgraph(Operation* root) {
  std::cout << root->getName().getStringRef().str() << std::endl;
  if (auto leaf = dyn_cast<LeafNodeInterface>(root)) {
    std::cout << "\tis a leaf." << std::endl;
  } else {
    std::cout << "\tis an inner node." << std::endl;
    for (auto op : root->getOperands()) {
      traverseSubgraph(op.getDefiningOp());
    }
  }

}*/


void SLPTree::buildGraph(std::vector<Operation*> const& operations, SLPNode& parentNode) {
  for (auto const& op : operations) {
    std::cout << op->getName().getStringRef().str() << std::endl;
  }
  // Stop growing graph
  if (!vectorizable(operations)) {
    return;
  }
  // Create new node for values and add to graph
  SLPNode* currentNode = &parentNode;
  // Recursion call to grow graph further
  // 1. Commutative
  if (commutative(operations)) {
    // A. Coarsening Mode
    for (auto const& operation : operations) {
      buildGraph(getOperands(operation), *currentNode);
    }
    // B. Normal Mode: Finished building multi-node
    if (currentNode->isMultiNode()) {
      reorderOperands(*currentNode);
      // TODO buildGraph()
    }
  }
    // 2. Non-Commutative
  else {
    buildGraph(getOperands(operations), *currentNode);
  }

}

/// Checks if the given operations are vectorizable. Operations are vectorizable iff the SPN dialect says they're
/// vectorizable and they all share the same opcode.
/// \param operations The potentially vectorizable operations.
/// \return True if the operations can be vectorized, otherwise false.
bool SLPTree::vectorizable(std::vector<Operation*> const& operations) const {
  for (size_t i = 0; i < operations.size(); ++i) {
    if (!operations.at(i)->hasTrait<OpTrait::spn::Vectorizable>()
        || (i > 0 && operations.at(i)->getName() != operations.front()->getName())) {
      return false;
    }
  }
  return true;
}

bool SLPTree::commutative(std::vector<Operation*> const& operations) const {
  return std::all_of(std::begin(operations), std::end(operations), [&](Operation* operation) {
    return operation->hasTrait<OpTrait::IsCommutative>();
  });
}

std::vector<Operation*> SLPTree::getOperands(std::vector<Operation*> const& values) const {
  std::vector<Operation*> operands;
  for (auto const& operation : values) {
    for (auto operand : operation->getOperands()) {
      operands.emplace_back(operand.getDefiningOp());
    }
  }
  return operands;
}

std::vector<Operation*> SLPTree::getOperands(Operation* operation) const {
  std::vector<Operation*> operands;
  for (auto operand : operation->getOperands()) {
    operands.emplace_back(operand.getDefiningOp());
  }
  return operands;
}

SLPTree::MODE SLPTree::modeFromOperation(Operation const* operation) const {
  if (dyn_cast<ConstantOp>(operation)) {
    return MODE::CONST;
  } else if (dyn_cast<HistogramOp>(operation) || dyn_cast<GaussianOp>(operation)) {
    return MODE::LOAD;
  }
  return MODE::OPCODE;
}

std::vector<SLPNode> SLPTree::reorderOperands(SLPNode& multinode) {
  assert(multinode.isMultiNode());
  std::vector<SLPNode> finalOrder;
  /*std::vector<MODE> mode;
  // 1. Strip first lane
  for (size_t i = 0; i < multinode.getOperands().size(); ++i) {
    auto& operand = multinode.getOperand(i);
    finalOrder.emplace_back(operand);
    mode.emplace_back(modeFromOperation(multinode.getOperation(0, i)));
  }

  // 2. For all other lanes, find best candidate
  for (size_t lane = 1; lane < multinode.numLanes(); ++lane) {
    std::vector<SLPNode*> candidates = multinode.getLane(lane);
    // Look for a matching candidate
    for (size_t i = 0; i < multinode.getOperands().size(); ++i) {
      // Skip if we can't vectorize
      if (mode.at(i) == MODE::FAILED) {
        continue;
      }
      auto const& last = finalOrder.at(i);
      auto const& bestResult = getBest(mode.at(i), last, candidates);

      // Update output
      // TODO: funky two dimensional stuff
      finalOrder.emplace_back(bestResult.first);

      // Detect SPLAT mode
      if (i == 1 && bestResult.first == last) {
        mode.at(i) = MODE::SPLAT;
      }

    }
  }*/
  return finalOrder;
}

std::pair<SLPNode*, SLPTree::MODE> SLPTree::getBest(SLPTree::MODE& mode,
                                                    SLPNode& last,
                                                    std::vector<SLPNode*>& candidates) const {
  SLPNode* best = nullptr;
  std::vector<SLPNode*> bestCandidates;
  if (mode == MODE::FAILED) {
    // Don't select now, let others choose first
    best = nullptr;
  } else if (mode == MODE::SPLAT) {
    // Look for other splat candidates
    for (auto const& value : candidates) {
      if (*value == last) {
        best = value;
        break;
      }
    }
  } else {
    // Default value
    best = candidates.front();
    for (auto const& candidate : candidates) {
      // We don't have consecutive memory accesses, therefore we're only interested in opcode comparisons.
      if (candidate->name() == last.name()) {
        bestCandidates.emplace_back(candidate);
      }
    }
    // 1. If we have a trivial solution, use it
    if (bestCandidates.empty()) {
      mode = MODE::FAILED;
    }
      // Single match
    else if (bestCandidates.size() == 1) {
      best = bestCandidates.front();
    }
      // 2. Look-ahead to choose from best candidates
    else {
      if (mode == MODE::OPCODE) {
        // Look-ahead on various levels
        for (size_t level = 1; level <= maxLookAhead; ++level) {
          // Best is the candidate with max score
          int bestScore = 0;
          std::set<int> scores;
          for (auto const& candidate : bestCandidates) {
            // Get the look-ahead score
            auto const& score = getLookAheadScore(last, *candidate, level);
            if (scores.empty() || score > bestScore) {
              best = candidate;
              bestScore = score;
              scores.insert(score);
            }
          }
          // If found best at level don't go deeper
          if (best != nullptr && scores.size() > 1) {
            break;
          }
        }
      }
    }

  }
  // Remove best from candidates
  if (best != nullptr) {
    candidates.erase(std::remove(std::begin(candidates), std::end(candidates), best), std::end(candidates));
  }
  return {best, mode};
}
int SLPTree::getLookAheadScore(SLPNode& last, SLPNode& candidate, size_t const& maxLevel) const {
  if (maxLevel == 0 || last.name() != candidate.name()) {
    return 0;
  }
  auto scoreSum = 0;
  for (size_t lane = 0; lane < last.numLanes(); ++lane) {
    for (auto& lastOperand : last.getOperands(lane)) {
      for (auto& candidateOperand : candidate.getOperands(lane)) {
        scoreSum += getLookAheadScore(lastOperand, candidateOperand, maxLevel - 1);
      }
    }
  }
  return scoreSum;
}
