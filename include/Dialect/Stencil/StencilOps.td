#ifndef STENCIL_OPS
#define STENCIL_OPS

include "Dialect/Stencil/StencilBase.td"
include "Dialect/Stencil/StencilInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Concrete Operations
//===----------------------------------------------------------------------===//

def Stencil_CastOp : Stencil_Op<"cast", [
  DeclareOpInterfaceMethods<ShapeOp>]> {
  let summary = "cast the dynamiclly shaped input fields";
  let description = [{
    This operation casts dynamically shaped input fields to statically shaped fields.

    Example:
      %0 = stencil.cast %in ([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  }];

  let arguments = (ins Stencil_Field:$field, Stencil_Index:$lb, Stencil_Index:$ub);
  let results = (outs Stencil_Field:$res);

  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<(ins "Value":$field, "ArrayRef<int64_t>":$lb, "ArrayRef<int64_t>":$ub), 
    [{
      $_state.addOperands(field);
      $_state.addAttribute(getLBAttrName(), $_builder.getI64ArrayAttr(lb));
      $_state.addAttribute(getUBAttrName(), $_builder.getI64ArrayAttr(ub));
      $_state.addTypes(stencil::FieldType::get(
        field.getType().cast<stencil::GridType>().getElementType(), 
        applyFunElementWise(lb, ub, std::minus<int64_t>())));
    }]>
  ];

  let assemblyFormat = [{
    $field `(` $lb `:` $ub `)` attr-dict-with-keyword `:` functional-type($field, $res)
  }];

  let hasVerifier = 1;
  
  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
  }];
}

def Stencil_IndexOp : Stencil_Op<"index", [
    DeclareOpInterfaceMethods<OffsetOp>,
    DeclareOpInterfaceMethods<ShiftOp>,
    Pure]> {
  let summary = "stencil index operation";
  let description = [{
    This operation returns the index of the current loop iteration for the
    chosen direction (0, 1, or 2).
    The offset is specified relative to the current position.

    Example:
      %0 = stencil.index 0 [-1, 0, 0] : index
  }];

  let arguments = (ins ConfinedAttr<I64Attr, [IntMinValue<0>, IntMaxValue<2>]>:$dim,
                     Stencil_Index:$offset);
  let results = (outs Index:$idx);

  let builders = [
    OpBuilder<(ins "int64_t":$dim, "ArrayRef<int64_t>":$offset), 
    [{
      $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));
      $_state.addAttribute("offset", $_builder.getI64ArrayAttr(offset));
      $_state.addTypes($_builder.getIndexType());
    }]>
  ];

  let assemblyFormat = [{
    $dim $offset attr-dict-with-keyword `:` type($idx)
  }];

  let hasVerifier = 1;
}

def Stencil_AccessOp : Stencil_Op<"access", [
    DeclareOpInterfaceMethods<OffsetOp>,
    DeclareOpInterfaceMethods<ShiftOp>,
    DeclareOpInterfaceMethods<ExtentOp>,
    Pure]> {
  let summary = "stencil access operation";
  let description = [{
    This operation accesses a temporary element given a constant 
    offset. The offset is specified relative to the current position.

    Example:
      %0 = stencil.access %temp [-1, 0, 0] : !stencil.temp<?x?x?xf64> -> f64
  }];

  let arguments = (ins Stencil_Temp:$temp, Stencil_Index:$offset);
  let results = (outs Stencil_Element:$res);
  
  let builders = [
    OpBuilder<(ins "Value":$temp, "ArrayRef<int64_t>":$offset), 
    [{
      $_state.addOperands(temp);
      $_state.addAttribute("offset", $_builder.getI64ArrayAttr(offset));
      auto tempType = temp.getType().cast<stencil::GridType>();
      $_state.addTypes(tempType.getElementType());
    }]>
  ];

  let assemblyFormat = [{
    $temp $offset attr-dict-with-keyword `:` functional-type($temp, $res)
  }];

  let hasVerifier = 1;
}

def Stencil_DynAccessOp : Stencil_Op<"dyn_access", [
  DeclareOpInterfaceMethods<ShiftOp, ["shiftByOffset"]>,
  DeclareOpInterfaceMethods<ExtentOp, ["getAccessExtent"]>,
  Pure]> {
  let summary = "stencil dyn_access operation";
  let description = [{
    This operation accesses a temporary element given a dynamic offset. 
    The offset is specified in absolute coordinates. An additional 
    range attribute specifies the maximal access extent relative to the
    iteration domain of the parent apply operation. 

    Example:
      %0 = stencil.dyn_access %temp (%i, %j, %k) in [-1, -1, -1] : [1, 1, 1] : !stencil.temp<?x?x?xf64> -> f64
  }];

  let arguments = (ins Stencil_Temp:$temp, 
                       Variadic<Index>:$offset, 
                       Stencil_Index:$lb, 
                       Stencil_Index:$ub);
  let results = (outs Stencil_Element:$res);
  
  let builders = [
    OpBuilder<(ins "Value":$temp, "ValueRange":$offset, "ArrayRef<int64_t>":$lb, "ArrayRef<int64_t>":$ub), 
    [{
      $_state.addOperands(temp);
      $_state.addOperands(offset);
      $_state.addAttribute(getLBAttrName(), $_builder.getI64ArrayAttr(lb));
      $_state.addAttribute(getUBAttrName(), $_builder.getI64ArrayAttr(ub));
      auto tempType = temp.getType().cast<stencil::GridType>();
      $_state.addTypes(tempType.getElementType());
    }]>
  ];

  let assemblyFormat = [{
    $temp `(` $offset `)` `in` $lb `:` $ub attr-dict-with-keyword `:` functional-type($temp, $res)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
  }];
}

def Stencil_LoadOp : Stencil_Op<"load", [
  DeclareOpInterfaceMethods<ShapeOp>,
  Pure]> {
  let summary = "load operation";
  let description = [{
    This operation takes a field and returns a temporary values.

    Example:
      %0 = stencil.load %field : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
  }];

  let arguments = (ins Stencil_Field:$field,
                       OptionalAttr<Stencil_Index>:$lb, 
                       OptionalAttr<Stencil_Index>:$ub);
  let results = (outs Stencil_Temp:$res);
  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<(ins "Value":$field, "std::optional<ArrayAttr>":$lb, "std::optional<ArrayAttr>":$ub), 
    [{
      $_state.addOperands(field);
      auto fieldType = field.getType().cast<stencil::GridType>();
      if(lb.has_value() && ub.has_value()) {
        $_state.addAttribute(getLBAttrName(), lb.value());
        $_state.addAttribute(getUBAttrName(), ub.value()); 
        Index lower, upper;
        for (auto &elem : lb.value()) 
          lower.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
        for (auto &elem : ub.value()) 
          upper.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
        $_state.addTypes(stencil::TempType::get(fieldType.getElementType(), fieldType.getAllocation(), lower, upper));
      } else {
        $_state.addTypes(stencil::TempType::get(fieldType.getElementType(), fieldType.getShape()));
      }
    }]>
  ];

  let assemblyFormat = [{
    $field (`(` $lb^ `:` $ub `)`)? attr-dict-with-keyword `:` functional-type($field, $res)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
  }];
}

def Stencil_BufferOp : Stencil_Op<"buffer", 
  [DeclareOpInterfaceMethods<ShapeOp>]> {
  let summary = "buffer operation";
  let description = [{
    Prevents fusion of consecutive stencil.apply operations.

    Example:
      %0 = stencil.buffer %buffered : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  }];

  let arguments = (ins Stencil_Temp:$temp,
                       OptionalAttr<Stencil_Index>:$lb,
                       OptionalAttr<Stencil_Index>:$ub);
  let results = (outs Stencil_Temp:$res);

  let builders = [
    OpBuilder<(ins "Value":$temp), 
    [{
      $_state.addOperands(temp);
      $_state.addTypes(temp.getType());
    }]>
  ];

  let assemblyFormat = [{
    $temp (`(` $lb^ `:` $ub`)`)? attr-dict-with-keyword `:` functional-type($temp, $res)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
  }];
}

def Stencil_StoreOp : Stencil_Op<"store", [
  DeclareOpInterfaceMethods<ShapeOp>]> {
  let summary = "store operation";
  let description = [{
    This operation takes a temp and writes a field on a user defined range.

    Example:
      stencil.store %temp to %field ([0,0,0] : [64,64,60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  }];

  let arguments = (ins Stencil_Temp:$temp, 
                       Stencil_Field:$field, 
                       Stencil_Index:$lb, 
                       Stencil_Index:$ub);
  let results = (outs);
  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<(ins "Value":$temp, "Value":$field, "ArrayRef<int64_t>":$lb, "ArrayRef<int64_t>":$ub), 
    [{
      $_state.addOperands({temp, field});
      $_state.addAttribute(getLBAttrName(), $_builder.getI64ArrayAttr(lb));
      $_state.addAttribute(getUBAttrName(), $_builder.getI64ArrayAttr(ub)); 
    }]>
  ];
  
  let assemblyFormat = [{
    $temp `to` $field `(` $lb `:` $ub `)` attr-dict-with-keyword `:` type($temp) `to` type($field)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
  }];
}

def Stencil_ApplyOp : Stencil_Op<"apply", [
  DeclareOpInterfaceMethods<ShapeOp, ["updateArgumentTypes"]>,
  IsolatedFromAbove, 
  SingleBlockImplicitTerminator<"ReturnOp">,
  Pure]> {
  let summary = "apply operation";
  let description = [{
    This operation takes a stencil function plus parameters and applies 
    the stencil function to the output temp.

    Example:

      %0 = stencil.apply (%arg0=%0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
        ...
      } 
  }];

  let arguments = (ins Variadic<AnyType>:$operands,
                        OptionalAttr<Stencil_Index>:$lb, 
                        OptionalAttr<Stencil_Index>:$ub);
  let results = (outs Variadic<Stencil_Temp>:$res);
  let regions = (region SizedRegion<1>:$region);
  let hasCanonicalizer = 1;
  
  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "ValueRange":$operands, "std::optional<ArrayAttr>":$lb, "std::optional<ArrayAttr>":$ub), 
    [{
      $_state.addOperands(operands);
      if(lb.has_value() && ub.has_value()) {
        $_state.addAttribute(getLBAttrName(), lb.value());
        $_state.addAttribute(getUBAttrName(), ub.value()); 
      }
      auto region = $_state.addRegion(); 
      region->push_back(new Block());

      llvm::SmallVector<mlir::Location> opLocs;
      opLocs.reserve(operands.size());

      for(auto operand : operands){
        opLocs.push_back(operand.getLoc());
      } 

      region->front().addArguments(operands.getTypes(), opLocs);
      $_state.addTypes(resultTypes);
    }]>,
    OpBuilder<(ins "TypeRange":$resultTypes, "std::optional<ArrayAttr>":$lb, "std::optional<ArrayAttr>":$ub), 
    [{
      if(lb.has_value() && ub.has_value()) {
        $_state.addAttribute(getLBAttrName(), lb.value());
        $_state.addAttribute(getUBAttrName(), ub.value()); 
      }
      $_state.addRegion();
      $_state.addTypes(resultTypes);
    }]>
  ];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    bool hasOnlyEmptyStores();
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
    Block *getBody() { return &getRegion().front(); }
    ShapeOp getCombineTreeRootShape();
  }];
}

def Stencil_StoreResultOp : Stencil_Op<"store_result", 
  [Pure]> {
  let summary = "store_result operation";
  let description = [{
    The store_result operation either stores an operand value or nothing.

    Examples:
      stencil.store_result %0 : !stencil.result<f64>
      stencil.store_result : !stencil.result<f64>      
  }];

  let arguments = (ins Variadic<Stencil_Element>:$operands);
  let results = (outs Stencil_Result:$res);

  let builders = [
    OpBuilder<(ins "Value":$value), 
    [{
      $_state.addOperands({value});
      $_state.addTypes(stencil::ResultType::get(value.getType()));
    }]>
  ];

  let assemblyFormat = [{
    ($operands^)? attr-dict-with-keyword `:` functional-type($operands, $res)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    std::optional<SmallVector<OpOperand *, 10>> getReturnOpOperands();
  }];
}

def Stencil_ReturnOp : Stencil_Op<"return", [
  Terminator,
  HasParent<"ApplyOp">]> {
  let summary = "return operation";
  let description = [{
    The return operation terminates the the stencil apply and writes
    the results of the stencil operator to the temporary values returned
    by the stencil apply operation. The types and the number of operands
    must match the results of the stencil apply operation.

    The optional unroll attribute enables the implementation of loop
    unrolling at the stencil dialect level.

    Examples:
      stencil.return %0 : !stencil.result<f64>
  }];

  let arguments = (ins Variadic<Stencil_Result>:$operands,
                   OptionalAttr<Stencil_Index>:$unroll);

  let builders = [
    OpBuilder<(ins "ValueRange":$operands, "std::optional<ArrayAttr>":$unroll), 
    [{
      $_state.addOperands({operands});
      if(unroll.has_value())
        $_state.addAttribute("unroll", unroll.value());
    }]>
  ];

  let assemblyFormat = [{
    (`unroll` $unroll^)? $operands attr-dict-with-keyword `:` type($operands)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    Index getMyUnroll() {
      Index result;
      ArrayAttr unrollAttr = getUnroll().value();
      for (auto &elem : unrollAttr) 
        result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
      return result;
    }
    unsigned getUnrollFac() {
      unsigned factor = 1;
      if (getUnroll().has_value()) {
        Index unroll = getMyUnroll();
        factor = std::accumulate(unroll.begin(), unroll.end(), 1,
                                      std::multiplies<int64_t>());
      }
      return factor;
    }
    size_t getUnrollDim() {
      size_t dimension = 0;
      if (getUnroll().has_value()) {
        auto unroll = getMyUnroll();
        auto unrollFac = getUnrollFac();
        auto it = llvm::find_if(unroll, [&](int64_t x) {
          return x == unrollFac;
        });
        assert(it != unroll.end() &&
              "expected to find unroll factor");
        dimension = std::distance(unroll.begin(), it);
      }
      return dimension;
    } 
  }];
}

def Stencil_CombineOp : Stencil_Op<"combine", [
  DeclareOpInterfaceMethods<ShapeOp>, 
  AttrSizedOperandSegments]> {
  let summary = "combine operation";
  let description = [{
    Combines the results computed on a lower with the results computed on 
    an upper domain. The operation combines the domain at a given index/offset 
    in a given dimension. Optional extra operands allow to combine values
    that are only written / defined on the lower or upper subdomain. The result
    values have the order upper/lower, lowerext, upperext.

    Example:
      %result = stencil.combine 2 at 11 lower = (%0 : !stencil.temp<?x?x?xf64>) upper = (%1 : !stencil.temp<?x?x?xf64>) lowerext = (%2 : !stencil.temp<?x?x?xf64>): !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>
  }];

  let arguments = (ins ConfinedAttr<I64Attr, [IntMinValue<0>, IntMaxValue<2>]>:$dim,
                       I64Attr:$index,
                       Variadic<Stencil_Temp>:$lower,
                       Variadic<Stencil_Temp>:$upper,
                       Variadic<Stencil_Temp>:$lowerext,
                       Variadic<Stencil_Temp>:$upperext,
                       OptionalAttr<Stencil_Index>:$lb,
                       OptionalAttr<Stencil_Index>:$ub);
  let results = (outs Variadic<Stencil_Temp>:$res);
  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    $dim `at` $index `lower` `=` `(`$lower `:` type($lower)`)` `upper` `=` `(`$upper `:` type($upper)`)` (`lowerext` `=` `(`$lowerext^ `:` type($lowerext)`)`)? (`upperext` `=` `(`$upperext^ `:` type($upperext)`)`)? (`(` $lb^ `:` $ub`)`)? attr-dict-with-keyword `:` type($res)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    int64_t getMyIndex() { return getIndexAttr().getValue().getSExtValue(); }
    SmallVector<Operation*, 4> getLowerDefiningOps() {
      SmallVector<Operation*, 4> definingOps;
      for(auto value : getLower()) {
        if(!llvm::is_contained(definingOps, value.getDefiningOp()))
          definingOps.push_back(value.getDefiningOp());
      }
      for(auto value : getLowerext()) {
        if(!llvm::is_contained(definingOps, value.getDefiningOp()))
          definingOps.push_back(value.getDefiningOp());
      }
      return definingOps;  
    }
    SmallVector<Operation*, 4> getUpperDefiningOps() {
      SmallVector<Operation*, 4> definingOps;
      for(auto value : getUpper()) {
        if(!llvm::is_contained(definingOps, value.getDefiningOp()))
          definingOps.push_back(value.getDefiningOp());
      }
      for(auto value : getUpperext()) {
        if(!llvm::is_contained(definingOps, value.getDefiningOp()))
          definingOps.push_back(value.getDefiningOp());
      }
      return definingOps;  
    }
    std::optional<unsigned> getLowerOperandNumber(unsigned resultNumber) {
      if(resultNumber < getLower().size()) {
        return resultNumber;
      }
      return {};
    }
    std::optional<unsigned> getLowerExtraOperandNumber(unsigned resultNumber) {
      if(resultNumber >= getLower().size() && resultNumber < getLower().size() + getLowerext().size()) {
        return resultNumber - getLower().size();
      }
      return {};
    }
    std::optional<unsigned> getUpperExtraOperandNumber(unsigned resultNumber) {
      if(resultNumber >= getLower().size() + getLowerext().size() && resultNumber < getLower().size() + getLowerext().size() + getUpperext().size()) {
        return resultNumber - getLower().size() - getLowerext().size();
      }
      return {};
    }
    bool isLowerOperand(unsigned operandNumber) {
      return operandNumber < getLower().size() ||
        (operandNumber >= getLower().size() + getUpper().size() && 
         operandNumber < getLower().size() + getUpper().size() + getLowerext().size());
    }
    bool isUpperOperand(unsigned operandNumber) {
      return !isLowerOperand(operandNumber) && operandNumber < getNumOperands();
    }
    static StringRef getLBAttrName() { return "lb"; }
    static StringRef getUBAttrName() { return "ub"; }
    stencil::CombineOp getCombineTreeRoot();
  }];
}

#endif // STENCIL_OPS
