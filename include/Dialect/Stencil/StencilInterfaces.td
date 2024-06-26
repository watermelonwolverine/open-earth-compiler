#ifndef Stencil_INTERFACES
#define Stencil_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Stencil Interfaces
//===----------------------------------------------------------------------===//

def ShapeOp : OpInterface<"ShapeOp"> {
  let description = [{
    Interface to get the operation bounds.
  }];

  let methods = [
    InterfaceMethod<"/*Get the lower bound of the operation*/",
      "Index", "getLB", (ins), [{
        Index result;
        std::optional<ArrayAttr> lb = $_op.getLb();
        for (auto &elem : lb.value()) 
          result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
        return result;
    }]>,
    InterfaceMethod<"/*Get the upper bound of the operation*/",
      "Index", "getUB", (ins), [{
        Index result;
        std::optional<ArrayAttr> ub = $_op.getUb();
        for (auto &elem : ub.value()) 
          result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
        return result;
    }]>,
    InterfaceMethod<"/*Clear the optional range attributes*/",
      "void", "clearInferredShape", (ins), [{
        if(isOptionalArrayAttr($_op.getLb()) && isOptionalArrayAttr($_op.getUb())) {
          $_op->removeAttr($_op.getLBAttrName());
          $_op->removeAttr($_op.getUBAttrName());
          // Update the result types
          for (auto result : $_op.getOperation()->getResults()) {
            auto oldType = result.getType().template cast<GridType>();
            result.setType(TempType::get(oldType.getElementType(), oldType.getAllocation()));
          }
        }
    }]>,
    InterfaceMethod<"/*Set the shape of the operation*/",
      "void", "updateShape", (ins "ArrayRef<int64_t>":$lb, "ArrayRef<int64_t>":$ub), [{
        // Update the shape
        SmallVector<Attribute, kIndexSize> lbAttr, ubAttr;
        llvm::transform(lb, std::back_inserter(lbAttr), [&](int64_t x) {
          return IntegerAttr::get(IntegerType::get($_op.getContext(), 64), x);
        });
        llvm::transform(ub, std::back_inserter(ubAttr), [&](int64_t x) {
          return IntegerAttr::get(IntegerType::get($_op.getContext(), 64), x);
        });
        $_op.setLbAttr(ArrayAttr::get($_op.getContext(), lbAttr));
        $_op.setUbAttr(ArrayAttr::get($_op.getContext(), ubAttr));
        // Update the result types
        for (auto result : $_op.getOperation()->getResults()) {
          auto oldType = result.getType().template cast<GridType>();
          result.setType(TempType::get(oldType.getElementType(), oldType.getAllocation(), lb, ub));
        }
    }]>,    
    InterfaceMethod<"/*Verify if the operation has valid bounds*/",
       "bool", "hasShape", (ins), [{
        std::optional<ArrayAttr> lb = $_op.getLb();
        std::optional<ArrayAttr> ub = $_op.getUb();
        return lb.has_value() && ub.has_value();
    }]>,   
    InterfaceMethod<"/*Get the rank of the operation*/",
      "unsigned", "getRank", (ins), [{
        std::optional<ArrayAttr> lb = $_op.getLb();
        std::optional<ArrayAttr> ub = $_op.getUb();
        assert(lb.value().size() == ub.value().size() && 
               "expected lower and upper bound to have the same rank");
        return (unsigned)lb.value().size();
    }]>,
    InterfaceMethod<"/*Update the argument types*/",
      "void", "updateArgumentTypes", (ins ), /*methodBody=*/[{}], [{
        // empty default implementation
    }]>,
  ];
}

def OffsetOp : OpInterface<"OffsetOp"> {
  let description = [{
    Interface to get the operation offset.
  }];

  let methods = [
    InterfaceMethod<"/*Get the offset of the operation*/",
      "Index", "getOffset", (ins), [{
        Index result;
        ArrayAttr offset = $_op.getOffset();
        for (auto &elem : offset)
          result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
        return result;
    }]>,
  ];
}

def ShiftOp : OpInterface<"ShiftOp"> {
  let description = [{
    Interface to shift the operation offset.
  }];

  let methods = [
    InterfaceMethod<"/*Shift operation by a constant offset*/",
      "void", "shiftByOffset", (ins "ArrayRef<int64_t>":$offset), [{}],
        /*defaultImplementation=*/[{
        SmallVector<Attribute, kIndexSize> offsetAttr;
        for (auto elem : llvm::zip(offset, $_op.getOffset())) {
          offsetAttr.push_back(IntegerAttr::get(
              IntegerType::get($_op.getContext(), 64),
              std::get<1>(elem).template cast<IntegerAttr>().getValue().getSExtValue() + 
              std::get<0>(elem)));
        }
        $_op.setOffsetAttr(ArrayAttr::get($_op.getContext(), offsetAttr));
    }]>,
  ];
}
def ExtentOp : OpInterface<"ExtentOp"> {
  let description = [{
    Interface to get the access extent.
  }];

  let methods = [
    InterfaceMethod<"/*Get the accessed temporary*/",
      "Value", "getTemp", (ins), [{
        return $_op.getTemp();
    }]>,
    InterfaceMethod<"/*Get the access extent*/",
      "std::tuple<Index, Index>", "getAccessExtent", (ins), [{}],
        /*defaultImplementation=*/[{
        Index result;
        ArrayAttr offset = $_op.getOffset();
        for (auto &elem : offset.getValue())
          result.push_back(elem.cast<IntegerAttr>().getValue().getSExtValue());
        return std::make_tuple(result, result);
    }]>,
  ];
}

#endif // Stencil_INTERFACES
