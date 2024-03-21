#ifndef CONVERSION_STENCILTOSTANDARD_CONVERTSTENCILTOSTANDARD
#define CONVERSION_STENCILTOSTANDARD_CONVERTSTENCILTOSTANDARD

include "mlir/Pass/PassBase.td"

def StencilToStandardPass : Pass<"convert-stencil-to-std", "ModuleOp"> {
  let summary = "Convert stencil dialect to standard operations";
  let constructor = "mlir::stencil::createConvertStencilToStandardPass()";

  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::memref::MemRefDialect",
    "mlir::gpu::GPUDialect",
  ];
}

#endif // CONVERSION_STENCILTOSTANDARD_CONVERTSTENCILTOSTANDARD
