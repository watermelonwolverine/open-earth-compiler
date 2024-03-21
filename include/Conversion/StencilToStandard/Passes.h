#ifndef CONVERSION_STENCILTOSTANDARD_PASSES_H
#define CONVERSION_STENCILTOSTANDARD_PASSES_H

#include "mlir/Pass/Pass.h"

// For Passes.h.inc
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {

class Pass;

namespace stencil {

std::unique_ptr<Pass> createConvertStencilToStandardPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Conversion/StencilToStandard/Passes.h.inc"

} // namespace stencil
} // namespace mlir

#endif // CONVERSION_STENCILTOSTANDARD_PASSES_H
