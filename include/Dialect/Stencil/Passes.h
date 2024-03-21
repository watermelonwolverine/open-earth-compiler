#ifndef DIALECT_STENCIL_PASSES_H
#define DIALECT_STENCIL_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace stencil {

std::unique_ptr<OperationPass<func::FuncOp>> createDomainSplitPass();

std::unique_ptr<OperationPass<func::FuncOp>> createStencilInliningPass();

std::unique_ptr<OperationPass<func::FuncOp>> createStencilUnrollingPass();

std::unique_ptr<OperationPass<func::FuncOp>> createCombineToIfElsePass();

std::unique_ptr<OperationPass<func::FuncOp>> createShapeInferencePass();

std::unique_ptr<OperationPass<func::FuncOp>> createShapeOverlapPass();

std::unique_ptr<OperationPass<func::FuncOp>> createStorageMaterializationPass();

std::unique_ptr<OperationPass<func::FuncOp>> createPeelOddIterationsPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Dialect/Stencil/Passes.h.inc"

} // namespace stencil
} // namespace mlir

#endif // DIALECT_STENCIL_PASSES_H
