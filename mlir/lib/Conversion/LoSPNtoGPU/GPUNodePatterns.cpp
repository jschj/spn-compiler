//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//

#include "LoSPNtoGPU/GPUNodePatterns.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "LoSPN/LoSPNAttributes.h"

mlir::LogicalResult mlir::spn::BatchReadGPULowering::matchAndRewrite(mlir::spn::low::SPNBatchRead op,
                                                                     llvm::ArrayRef<mlir::Value> operands,
                                                                     mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  // Replace the BatchRead with a scalar load from the input memref,
  // using the batchIndex and the constant sample index.
  assert(operands.size() == 2 && "Expecting two operands for BatchRead");
  assert(operands[0].getType().isa<MemRefType>());
  assert(operands[1].getType().isa<IndexType>());
  auto constSampleIndex = rewriter.create<ConstantOp>(op->getLoc(), rewriter.getIndexAttr(op.sampleIndex()));
  rewriter.replaceOpWithNewOp<LoadOp>(op, operands[0], ValueRange{operands[1], constSampleIndex});
  return success();
}

mlir::LogicalResult mlir::spn::BatchWriteGPULowering::matchAndRewrite(mlir::spn::low::SPNBatchWrite op,
                                                                      llvm::ArrayRef<mlir::Value> operands,
                                                                      mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  // Replace the BatchWrite with a store to the input memref,
  // using the batchIndex.
  assert(operands.size() == 3 && "Expecting three operands for BatchWrite");
  assert(operands[1].getType().isa<MemRefType>());
  assert(operands[1].getType().dyn_cast<MemRefType>().getElementType() == operands[0].getType()
             && "Result type and element type of MemRef must match");
  assert(operands[2].getType().isa<IndexType>());
  rewriter.replaceOpWithNewOp<StoreOp>(op, operands[0], operands[1], operands[2]);
  return success();
}

mlir::LogicalResult mlir::spn::CopyGPULowering::matchAndRewrite(mlir::spn::low::SPNCopy op,
                                                                llvm::ArrayRef<mlir::Value> operands,
                                                                mlir::ConversionPatternRewriter& rewriter) const {
  assert(operands.size() == 2 && "Expecting two operands for Copy");
  assert(operands[0].getType().isa<MemRefType>());
  assert(operands[1].getType().isa<MemRefType>());
  rewriter.replaceOpWithNewOp<linalg::CopyOp>(op, operands[0], operands[1]);
  return success();
}

// Anonymous namespace holding helper functions.
mlir::LogicalResult mlir::spn::ConstantGPULowering::matchAndRewrite(mlir::spn::low::SPNConstant op,
                                                                    llvm::ArrayRef<mlir::Value> operands,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  assert(operands.empty() && "Expecting no operands for Constant");
  Type resultType = op.getResult().getType();
  if (auto logType = resultType.dyn_cast<low::LogType>()) {
    resultType = logType.getBaseType();
  }
  FloatAttr value = op.valueAttr();
  if (resultType != rewriter.getF64Type()) {
    assert(resultType.isa<FloatType>());
    value = rewriter.getFloatAttr(resultType, value.getValueAsDouble());
  }
  rewriter.replaceOpWithNewOp<ConstantOp>(op, resultType, value);
  return success();
}

mlir::LogicalResult mlir::spn::ReturnGPULowering::matchAndRewrite(mlir::spn::low::SPNReturn op,
                                                                  llvm::ArrayRef<mlir::Value> operands,
                                                                  mlir::ConversionPatternRewriter& rewriter) const {
  if (!operands.empty()) {
    // At this point, all Tensor semantic should have been removed by the bufferization.
    // Hence, the SPNReturn, which can only return Tensors, should not have any return values anymore
    // and should merely be used as a terminator for Kernels and Tasks.
    return rewriter.notifyMatchFailure(op,
                                       "SPNReturn can only return Tensors, which should have been removed by bufferization");
  }
  rewriter.replaceOpWithNewOp<ReturnOp>(op);
  return success();
}

mlir::LogicalResult mlir::spn::LogGPULowering::matchAndRewrite(mlir::spn::low::SPNLog op,
                                                               llvm::ArrayRef<mlir::Value> operands,
                                                               mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  assert(operands.size() == 1 && "Expecting one operand for Log");
  rewriter.replaceOpWithNewOp<math::LogOp>(op, operands[0]);
  return success();
}

mlir::LogicalResult mlir::spn::MulGPULowering::matchAndRewrite(mlir::spn::low::SPNMul op,
                                                               llvm::ArrayRef<mlir::Value> operands,
                                                               mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  if (op.getResult().getType().isa<low::LogType>()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not match for log-space computation");
  }
  assert(operands.size() == 2 && "Expecting two operands for Mul");
  if (!operands[0].getType().isa<FloatType>()) {
    return rewriter.notifyMatchFailure(op, "Currently only matches floating-point multiplications");
  }
  rewriter.replaceOpWithNewOp<MulFOp>(op, operands[0], operands[1]);
  return success();
}

mlir::LogicalResult mlir::spn::MulLogGPULowering::matchAndRewrite(mlir::spn::low::SPNMul op,
                                                                  llvm::ArrayRef<mlir::Value> operands,
                                                                  mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  if (!op.getResult().getType().isa<low::LogType>()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches for log-space computation");
  }
  assert(operands.size() == 2 && "Expecting two operands for Mul");
  if (!operands[0].getType().isa<FloatType>()) {
    return rewriter.notifyMatchFailure(op, "Currently only matches floating-point multiplications");
  }
  rewriter.replaceOpWithNewOp<AddFOp>(op, operands[0], operands[1]);
  return success();
}

mlir::LogicalResult mlir::spn::AddGPULowering::matchAndRewrite(mlir::spn::low::SPNAdd op,
                                                               llvm::ArrayRef<mlir::Value> operands,
                                                               mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  if (op.getResult().getType().isa<low::LogType>()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not match for log-space computation");
  }
  assert(operands.size() == 2 && "Expecting two operands for Add");
  if (!operands[0].getType().isa<FloatType>()) {
    return rewriter.notifyMatchFailure(op, "Currently only matches floating-point additions");
  }
  rewriter.replaceOpWithNewOp<AddFOp>(op, operands[0], operands[1]);
  return success();
}

mlir::LogicalResult mlir::spn::AddLogGPULowering::matchAndRewrite(mlir::spn::low::SPNAdd op,
                                                                  llvm::ArrayRef<mlir::Value> operands,
                                                                  mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  if (!op.getResult().getType().isa<low::LogType>()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches for log-space computation");
  }
  assert(operands.size() == 2 && "Expecting two operands for Mul");
  if (!operands[0].getType().isa<FloatType>()) {
    return rewriter.notifyMatchFailure(op, "Currently only matches floating-point multiplications");
  }
  // Calculate addition 'x + y' in log-space as
  // 'a + log(1 + exp(b-a)', with a == log(x),
  // b == log(y) and a > b.
  auto compare = rewriter.create<CmpFOp>(op.getLoc(), CmpFPredicate::OGT, operands[0], operands[1]);
  auto a = rewriter.create<SelectOp>(op->getLoc(), compare, operands[0], operands[1]);
  auto b = rewriter.create<SelectOp>(op->getLoc(), compare, operands[1], operands[0]);
  auto sub = rewriter.create<SubFOp>(op->getLoc(), b, a);
  auto exp = rewriter.create<math::ExpOp>(op.getLoc(), sub);
  // TODO Currently, GPULowering of log1p to LLVM is not supported,
  // therefore we perform the computation manually here.
  auto constantOne = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getFloatAttr(operands[0].getType(), 1.0));
  auto onePlus = rewriter.create<AddFOp>(op->getLoc(), constantOne, exp);
  auto log = rewriter.create<math::LogOp>(op.getLoc(), onePlus);
  rewriter.replaceOpWithNewOp<AddFOp>(op, a, log);
  return success();
}

mlir::LogicalResult mlir::spn::GaussianGPULowering::matchAndRewrite(mlir::spn::low::SPNGaussianLeaf op,
                                                                    llvm::ArrayRef<mlir::Value> operands,
                                                                    mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  if (op.getResult().getType().isa<low::LogType>()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not match for log-space computation");
  }
  assert(operands.size() == 1 && "Expecting a single operands for Gaussian");
  if (!operands.front().getType().isIntOrFloat()) {
    return rewriter.notifyMatchFailure(op, "Pattern only takes int or float as input");
  }
  if (!op.getResult().getType().isa<FloatType>()) {
    return rewriter.notifyMatchFailure(op, "Cannot match Gaussian computing non-float result");
  }
  auto index = operands[0];
  auto resultType = op.getResult().getType().dyn_cast<FloatType>();

  auto indexType = index.getType();
  if (indexType.isIntOrIndex()) {
    // Convert integer/index input to floating point
    index = rewriter.create<UIToFPOp>(op->getLoc(), index, resultType);
  } else if (auto floatIndexType = indexType.dyn_cast<FloatType>()) {
    // Widden or narrow the index floating-point type to the result floating-point type.
    if (floatIndexType.getWidth() < resultType.getWidth()) {
      index = rewriter.create<mlir::FPExtOp>(op.getLoc(), index, resultType);
    } else if (floatIndexType.getWidth() > resultType.getWidth()) {
      index = rewriter.create<mlir::FPTruncOp>(op.getLoc(), index, resultType);
    }
  } else {
    // The input is neither float nor integer/index, fail this pattern because no conversion is possible.
    return rewriter.notifyMatchFailure(op, "Match failed because input is neither float nor integer/index");
  }

  // Calculate Gaussian distribution using e^(-(x - mean)^2/2*variance))/sqrt(2*PI*variance)
  // Variance from standard deviation.
  double variance = op.stddev().convertToDouble() * op.stddev().convertToDouble();
  // 1/sqrt(2*PI*variance)
  double coefficient = 1.0 / (std::sqrt(2.0 * M_PI * variance));
  auto coefficientConst = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getF64FloatAttr(coefficient));
  // -1/(2*variance)
  double denominator = -1.0 / (2.0 * variance);
  auto denominatorConst = rewriter.create<mlir::ConstantOp>(op.getLoc(), rewriter.getF64FloatAttr(denominator));
  // x - mean
  auto meanConst = rewriter.create<mlir::ConstantOp>(op.getLoc(), op.meanAttr());
  auto subtraction = rewriter.create<mlir::SubFOp>(op.getLoc(), index, meanConst);
  // (x-mean)^2
  auto numerator = rewriter.create<mlir::MulFOp>(op.getLoc(), subtraction, subtraction);
  // -(x-mean)^2 / 2*variance
  auto fraction = rewriter.create<mlir::MulFOp>(op.getLoc(), numerator, denominatorConst);
  // e^(-(x-mean)^2 / 2*variance)
  auto exp = rewriter.create<mlir::math::ExpOp>(op.getLoc(), fraction);
  // e^(-(x - mean)^2/2*variance)) * 1/sqrt(2*PI*variance)
  rewriter.replaceOpWithNewOp<mlir::MulFOp>(op, coefficientConst, exp);
  return success();
}

mlir::LogicalResult mlir::spn::GaussianLogGPULowering::matchAndRewrite(mlir::spn::low::SPNGaussianLeaf op,
                                                                       llvm::ArrayRef<mlir::Value> operands,
                                                                       mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not vectorize, no match");
  }
  if (!op.getResult().getType().isa<low::LogType>()) {
    return rewriter.notifyMatchFailure(op, "Pattern only matches for log-space computation");
  }
  assert(operands.size() == 1 && "Expecting a single operands for Gaussian");
  if (!operands.front().getType().isIntOrFloat()) {
    return rewriter.notifyMatchFailure(op, "Pattern only takes int or float as input");
  }
  if (!op.getResult().getType().cast<low::LogType>().getBaseType().isa<FloatType>()) {
    return rewriter.notifyMatchFailure(op, "Cannot match Gaussian computing non-float result");
  }
  auto index = operands[0];
  auto resultType = op.getResult().getType().cast<low::LogType>().getBaseType().dyn_cast<FloatType>();
  assert(resultType);

  auto indexType = index.getType();
  if (indexType.isIntOrIndex()) {
    // Convert integer/index input to floating point
    index = rewriter.create<UIToFPOp>(op->getLoc(), index, resultType);
  } else if (auto floatIndexType = indexType.dyn_cast<FloatType>()) {
    // Widden or narrow the index floating-point type to the result floating-point type.
    if (floatIndexType.getWidth() < resultType.getWidth()) {
      index = rewriter.create<mlir::FPExtOp>(op.getLoc(), index, resultType);
    } else if (floatIndexType.getWidth() > resultType.getWidth()) {
      index = rewriter.create<mlir::FPTruncOp>(op.getLoc(), index, resultType);
    }
  } else {
    // The input is neither float nor integer/index, fail this pattern because no conversion is possible.
    return rewriter.notifyMatchFailure(op, "Match failed because input is neither float nor integer/index");
  }

  // Calculate Gaussian distribution using the logarithm of the PDF of the Normal (Gaussian) distribution,
  // given as '-ln(stddev) - 1/2 ln(2*pi) - (x - mean)^2 / 2*stddev^2'
  // First term, -ln(stddev)
  double firstTerm = -log(op.stddev().convertToDouble());
  // Second term, - 1/2 ln(2*pi)
  double secondTerm = -0.5 * log(2 * M_PI);
  // Denominator, - 1/2*(stddev^2)
  double denominator = -(1.0 / (2.0 * op.stddev().convertToDouble() * op.stddev().convertToDouble()));
  auto denominatorConst = rewriter.create<mlir::ConstantOp>(op.getLoc(),
                                                            rewriter.getFloatAttr(resultType, denominator));
  // Coefficient, summing up the first two constant terms
  double coefficient = firstTerm + secondTerm;
  auto coefficientConst = rewriter.create<mlir::ConstantOp>(op->getLoc(),
                                                            rewriter.getFloatAttr(resultType, coefficient));
  // x - mean
  auto meanConst = rewriter.create<mlir::ConstantOp>(op.getLoc(),
                                                     rewriter.getFloatAttr(resultType,
                                                                           op.meanAttr().getValueAsDouble()));
  auto subtraction = rewriter.create<mlir::SubFOp>(op.getLoc(), index, meanConst);
  // (x-mean)^2
  auto numerator = rewriter.create<mlir::MulFOp>(op.getLoc(), subtraction, subtraction);
  // - ( (x-mean)^2 / 2 * stddev^2 )
  auto fraction = rewriter.create<mlir::MulFOp>(op.getLoc(), numerator, denominatorConst);
  // -ln(stddev) - 1/2 ln(2*pi) - 1/2*(stddev^2) * (x - mean)^2
  rewriter.replaceOpWithNewOp<mlir::AddFOp>(op, coefficientConst, fraction);
  return success();
}

mlir::LogicalResult mlir::spn::ResolveStripLogGPU::matchAndRewrite(mlir::spn::low::SPNStripLog op,
                                                                   llvm::ArrayRef<mlir::Value> operands,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
  if (op.checkVectorized()) {
    return rewriter.notifyMatchFailure(op, "Pattern does not resolve vectorized operation");
  }
  assert(operands.size() == 1);
  if (operands[0].getType() != op.target()) {
    return rewriter.notifyMatchFailure(op, "Could not resolve StripLog trivially");
  }
  rewriter.replaceOp(op, operands[0]);
  return success();
}