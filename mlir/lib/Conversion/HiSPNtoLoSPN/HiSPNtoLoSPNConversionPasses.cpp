//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include <HiSPNtoLoSPN/QueryPatterns.h>
#include "HiSPNtoLoSPN/ArithmeticPrecisionAnalysis.h"
#include "HiSPNtoLoSPN/HiSPNtoLoSPNConversionPasses.h"
#include "HiSPNtoLoSPN/NodePatterns.h"
#include "LoSPN/LoSPNDialect.h"
#include "HiSPN/HiSPNOps.h"
#include "HiSPNtoLoSPN/HiSPNTypeConverter.h"

using namespace mlir::spn;

void HiSPNtoLoSPNNodeConversionPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<high::HiSPNDialect>();
  target.addLegalDialect<low::LoSPNDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  target.addIllegalOp<high::ProductNode, high::SumNode, high::HistogramNode,
                      high::CategoricalNode, high::GaussianNode,
                      high::RootNode>();

  // Use type analysis to determine data type for actual computation.
  // The concrete type determined by the analysis replaces the abstract
  // probability type used by the HiSPN dialect.
  std::unique_ptr<HiSPNTypeConverter> typeConverter;
  if (optimizeRepresentation) {
    auto& arithmeticAnalysis = getAnalysis<mlir::spn::ArithmeticPrecisionAnalysis>();
    typeConverter = std::make_unique<HiSPNTypeConverter>(arithmeticAnalysis.getComputationType(computeLogSpace));
  } else if (computeLogSpace) {
    typeConverter =
        std::make_unique<HiSPNTypeConverter>(mlir::spn::low::LogType::get(mlir::FloatType::getF32(&getContext())));
  } else {
    typeConverter = std::make_unique<HiSPNTypeConverter>(mlir::Float64Type::get(&getContext()));
  }

  OwningRewritePatternList patterns;
  mlir::spn::populateHiSPNtoLoSPNNodePatterns(patterns, &getContext(), *typeConverter);

  auto op = getOperation();
  FrozenRewritePatternList frozenPatterns(std::move(patterns));
  if (failed(applyPartialConversion(op, target, frozenPatterns))) {
    signalPassFailure();
  }
  // Explicitly mark the ArithmeticPrecisionAnalysis as preserved, so the
  // QueryConversionPass can use the information, even though the Graph's
  // nodes have already been converted.
  markAnalysesPreserved<ArithmeticPrecisionAnalysis>();
}

std::unique_ptr<mlir::Pass> mlir::spn::createHiSPNtoLoSPNNodeConversionPass(bool useLogSpaceComputation,
                                                                            bool useOptimalRepresentation) {
  return std::make_unique<HiSPNtoLoSPNNodeConversionPass>(useLogSpaceComputation, useOptimalRepresentation);
}

void HiSPNtoLoSPNQueryConversionPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<low::LoSPNDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  target.addLegalOp<FuncOp>();

  target.addIllegalDialect<high::HiSPNDialect>();

  // Use type analysis to determine data type for actual computation.
  // The concrete type determined by the analysis replaces the abstract
  // probability type used by the HiSPN dialect.
  std::unique_ptr<HiSPNTypeConverter> typeConverter;
  if (optimizeRepresentation) {
    auto arithmeticAnalysis = getCachedAnalysis<ArithmeticPrecisionAnalysis>();
    assert(arithmeticAnalysis && "The arithmetic analysis needs to be preserved after node conversion");
    typeConverter = std::make_unique<HiSPNTypeConverter>(arithmeticAnalysis->get().getComputationType(computeLogSpace));
  } else if (computeLogSpace) {
    typeConverter =
        std::make_unique<HiSPNTypeConverter>(mlir::spn::low::LogType::get(mlir::FloatType::getF32(&getContext())));
  } else {
    typeConverter = std::make_unique<HiSPNTypeConverter>(mlir::Float64Type::get(&getContext()));
  }

  OwningRewritePatternList patterns;
  mlir::spn::populateHiSPNtoLoSPNQueryPatterns(patterns, &getContext(), *typeConverter);

  auto op = getOperation();
  FrozenRewritePatternList frozenPatterns(std::move(patterns));
  if (failed(applyFullConversion(op, target, frozenPatterns))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::spn::createHiSPNtoLoSPNQueryConversionPass(bool useLogSpaceComputation,
                                                                             bool useOptimalRepresentation) {
  return std::make_unique<HiSPNtoLoSPNQueryConversionPass>(useLogSpaceComputation, useOptimalRepresentation);
}
