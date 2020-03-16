//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#ifndef SPNC_COMPILER_SRC_CODEGEN_MLIR_MLIRBODYGEN_H
#define SPNC_COMPILER_SRC_CODEGEN_MLIR_MLIRBODYGEN_H

#include <mlir/IR/Builders.h>
#include <unordered_map>
#include <graph-ir/GraphIRNode.h>
#include <graph-ir/transform/BaseVisitor.h>
#include "mlir/IR/StandardTypes.h"

namespace spnc {

  using bucket_t = std::tuple<int, int, double>;

  ///
  /// Visitor generating SPN-dialect MLIR operations for each node in the SPN graph.
  class MLIRBodyGen : public BaseVisitor {

  public:

    /// Constructor.
    /// \param _builder MLIR operation builder used to create operations.
    /// \param n2v Mapping from graph-IR node's unique ID to MLIR value.
    MLIRBodyGen(mlir::OpBuilder* _builder, std::unordered_map<std::string, mlir::Value>* n2v);

    void visitHistogram(Histogram& n, arg_t arg) override;

    void visitProduct(Product& n, arg_t arg) override;

    void visitSum(Sum& n, arg_t arg) override;

    void visitWeightedSum(WeightedSum& n, arg_t arg) override;

  private:

    mlir::Value getValueForNode(NodeReference node, arg_t arg);

    mlir::OpBuilder* builder;

    std::unordered_map<std::string, mlir::Value>* node2value;
  };

}

#endif //SPNC_COMPILER_SRC_CODEGEN_MLIR_MLIRBODYGEN_H