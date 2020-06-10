#ifndef CONVERSION_LOOPSTOCUDA_PASSES_H
#define CONVERSION_LOOPSTOCUDA_PASSES_H

#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace mlir {

class Pass;

std::unique_ptr<Pass> createLaunchFuncToCUDACallsPass();

std::unique_ptr<OperationPass<FuncOp>> createStencilLoopMappingPass();

OwnedBlob compilePtxToCubin(const std::string &ptx, Location loc,
                             StringRef name);

void registerGPUToCUBINPipeline();
void registerGPUToHSACOPipeline();

} // namespace mlir

#endif // CONVERSION_LOOPSTOCUDA_PASSES_H
