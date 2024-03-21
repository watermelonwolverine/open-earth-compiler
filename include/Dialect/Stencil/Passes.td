#ifndef DIALECT_STENCIL_PASSES
#define DIALECT_STENCIL_PASSES

include "mlir/Pass/PassBase.td"

def DomainSplitPass : Pass<"stencil-domain-split", "func::FuncOp"> {
  let summary = "Copy apply ops used in multiple domains";
  let constructor = "mlir::stencil::createDomainSplitPass()";
}

def StencilInliningPass : Pass<"stencil-inlining", "func::FuncOp"> {
  let summary = "Inline stencil apply ops";
  let constructor = "mlir::stencil::createStencilInliningPass()";
}

def StencilUnrollingPass : Pass<"stencil-unrolling", "func::FuncOp"> {
  let summary = "Unroll stencil apply ops";
  let constructor = "mlir::stencil::createStencilUnrollingPass()";
  let options = [
    Option<"unrollFactor", "unroll-factor", "unsigned", /*default=*/"2",
           "Number of unrolled loop iterations">,
    Option<"unrollIndex", "unroll-index", "unsigned", /*default=*/"1",
           "Unroll index specifying the unrolling dimension">,
  ];
}

def CombineToIfElsePass : Pass<"stencil-combine-to-ifelse", "func::FuncOp"> {
  let summary = "Lower stencil combine ops to if/else";
  let constructor = "mlir::stencil::createCombineToIfElsePass()";
  let options = [
    Option<"prepareOnly", "prepare-only", "bool", /*default=*/"false", 
           "Lower extra operands and fuse multiple producers attached to one combine">,
    Option<"internalOnly", "internal-only", "bool", /*default=*/"false", 
           "Lower only combine ops embedded in between apply ops">,
  ];
}

def ShapeInferencePass : Pass<"stencil-shape-inference", "func::FuncOp"> {
  let summary = "Infer loop bounds and storage shapes";
  let constructor = "mlir::stencil::createShapeInferencePass()";
  let options = [
    Option<"extendStorage", "extend-storage", "bool", /*default=*/"false", 
           "Extend the shape of store and buffer ops">,
  ];
}

def ShapeOverlapPass : Pass<"stencil-shape-overlap",  "func::FuncOp"> {
  let summary = "Handle overlapping stores";
  let constructor = "mlir::stencil::createShapeOverlapPass()";
}

def StorageMaterializationPass : Pass<"stencil-storage-materialization", "func::FuncOp"> {
  let summary = "Introduce explicit storage between combine and apply ops";
  let constructor = "mlir::stencil::createStorageMaterializationPass()";
}

def PeelOddIterationsPass : Pass<"stencil-peel-odd-iterations", "func::FuncOp"> {
  let summary = "Peel odd iterations that are not a multiple of the unroll factor";
  let constructor = "mlir::stencil::createPeelOddIterationsPass()";
}

#endif // DIALECT_STENCIL_PASSES
