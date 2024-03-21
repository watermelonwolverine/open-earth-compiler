#include "Dialect/Stencil/Passes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/StencilUtils.h"
#include "PassDetail.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <cstdint>

using namespace mlir;
using namespace stencil;

namespace {

// Base class of all storage materialization patterns
template <typename SourceOp>
struct StorageMaterializationPattern : public OpRewritePattern<SourceOp> {
  StorageMaterializationPattern(MLIRContext *context,
                                PatternBenefit benefit = 1)
      : OpRewritePattern<SourceOp>(context, benefit) {}

  // Buffer all outputs connected to an apply but not to a store op
  bool doesEdgeRequireBuffering(Value value) const {
    if (llvm::any_of(value.getUsers(),
                     [](Operation *op) { return isa<stencil::ApplyOp>(op); }) &&
        llvm::none_of(value.getUsers(),
                      [](Operation *op) { return isa<stencil::StoreOp>(op); }))
      return true;
    return false;
  }

  // Buffer the results of the cloned operation and replace the matched op
  LogicalResult introduceResultBuffers(Operation *matchedOp,
                                       Operation *clonedOp,
                                       PatternRewriter &rewriter) const {
    SmallVector<Value, 10> repResults = clonedOp->getResults();
    for (auto result : matchedOp->getResults()) {
      if (doesEdgeRequireBuffering(result)) {
        auto bufferOp = rewriter.create<stencil::BufferOp>(
            matchedOp->getLoc(), result.getType(),
            clonedOp->getResult(result.getResultNumber()), nullptr, nullptr);
        repResults[result.getResultNumber()] = bufferOp;
      }
    }
    rewriter.replaceOp(matchedOp, repResults);
    return success();
  }
};

// Pattern introducing buffers between consecutive apply ops
struct ApplyOpRewrite : public StorageMaterializationPattern<stencil::ApplyOp> {
  using StorageMaterializationPattern<
      stencil::ApplyOp>::StorageMaterializationPattern;

  LogicalResult matchAndRewrite(stencil::ApplyOp applyOp,
                                PatternRewriter &rewriter) const override {
    if (llvm::any_of(applyOp.getResults(), [&](Value value) {
          return doesEdgeRequireBuffering(value);
        })) {
      // Clone the apply op and move the body
      auto clonedOp = rewriter.cloneWithoutRegions(applyOp);
      rewriter.inlineRegionBefore(applyOp.getRegion(), clonedOp.getRegion(),
                                  clonedOp.getRegion().begin());

      // Introduce a buffer on every result connected to another apply
      // TODO check
      return introduceResultBuffers(applyOp, clonedOp, rewriter);

    }
    return failure();
  }
};

// Pattern introducing buffers between consecutive apply ops
struct CombineOpRewrite
    : public StorageMaterializationPattern<stencil::CombineOp> {
  using StorageMaterializationPattern<
      stencil::CombineOp>::StorageMaterializationPattern;

  LogicalResult matchAndRewrite(stencil::CombineOp combineOp,
                                PatternRewriter &rewriter) const override {
    if (llvm::any_of(combineOp.getResults(), [&](Value value) {
          return doesEdgeRequireBuffering(value);
        })) {
      // Clone the combine op
      auto clonedOp = rewriter.clone(*combineOp.getOperation());

      // Introduce a buffer on every result connected to another apply
      // TODO check
      return introduceResultBuffers(combineOp, clonedOp, rewriter);
    }
    return failure();
  }
};

struct StorageMaterializationPass
    : public StorageMaterializationPassBase<StorageMaterializationPass> {

  void runOnOperation() override;
};

void StorageMaterializationPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  // Only run on functions marked as stencil programs
  if (!StencilDialect::isStencilProgram(funcOp))
    return;

  // Poppulate the pattern list depending on the config
  RewritePatternSet patterns(&getContext());
  patterns.insert<ApplyOpRewrite, CombineOpRewrite>(&getContext());
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-value"
  applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  #pragma clang diagnostic pop
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::stencil::createStorageMaterializationPass() {
  return std::make_unique<StorageMaterializationPass>();
}
