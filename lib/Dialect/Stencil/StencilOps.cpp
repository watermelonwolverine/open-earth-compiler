#include "Dialect/Stencil/StencilOps.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/StencilUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace mlir;
using namespace stencil;

//===----------------------------------------------------------------------===//
// stencil.apply
//===----------------------------------------------------------------------===//

LogicalResult CastOp::verify() {
  auto fieldType = getField().getType().cast<stencil::GridType>();
  auto resType = getRes().getType().cast<stencil::GridType>();
  auto shapeOp = cast<ShapeOp>(this->getOperation());
  if (!fieldType.hasDynamicShape())
    return emitOpError("expected field to have dynamic shape");
  if (resType.hasDynamicShape())
    return emitOpError("expected result to have static shape");
  if (fieldType.getAllocation() != resType.getAllocation())
    return emitOpError(
        "expected the field and result types to have the same allocation");
  if (fieldType.getElementType() != resType.getElementType())
    return emitOpError(
        "the field and result types have different element types");
  if (shapeOp.getRank() != fieldType.getRank() ||
      shapeOp.getRank() != resType.getRank())
    return emitOpError(
        "expected op and the field and result types to have the same rank");

  // Ensure the shape matches the result type
  if (!resType.hasEqualShape(shapeOp.getLB(), shapeOp.getUB()))
    return emitOpError("expected op and result type to have the same shape");

  // Verify all users fit the shape
  for (auto user : getRes().getUsers()) {
    if (auto userOp = dyn_cast<ShapeOp>(user)) {
      if (userOp.hasShape() &&
          (shapeOp.getLB() !=
               applyFunElementWise(shapeOp.getLB(), userOp.getLB(), min) ||
           shapeOp.getUB() !=
               applyFunElementWise(shapeOp.getUB(), userOp.getUB(), max)))
        return emitOpError("shape not large enough to fit all accesses");
    }
  }

  return success();
}

LogicalResult IndexOp::verify() { return success(); }

LogicalResult AccessOp::verify() {
  auto tempType = getTemp().getType().cast<stencil::GridType>();
  if (getOffset().size() != tempType.getRank())
    return emitOpError("offset and temp dimensions do not match");
  if (getRes().getType() != tempType.getElementType())
    return emitOpError("result type and element type are inconsistent");
  return success();
}

LogicalResult DynAccessOp::verify() {
  auto tempType = getTemp().getType().cast<stencil::GridType>();
  if (getOffset().size() != tempType.getRank())
    return emitOpError("offset and temp dimensions do not match");
  if (getRes().getType() != tempType.getElementType())
    return emitOpError("result type and element type are inconsistent");
  return success();
}

LogicalResult LoadOp::verify() {
  // Check the field and result types
  auto fieldType = getField().getType().cast<stencil::GridType>();
  auto resType = getRes().getType().cast<stencil::GridType>();
  if (fieldType.hasDynamicShape())
    return emitOpError("expected fields to have static shape");
  if (fieldType.getRank() != resType.getRank())
    return emitOpError("the field and temp types have different rank");
  if (fieldType.getAllocation() != resType.getAllocation())
    return emitOpError("the field and temp types have different allocation");
  if (fieldType.getElementType() != resType.getElementType())
    return emitOpError("the field and temp types have different element types");

  // Ensure the shape matches the field and result types
  auto shapeOp = cast<ShapeOp>(this->getOperation());
  if (shapeOp.hasShape()) {
    if (!fieldType.hasLargerOrEqualShape(shapeOp.getLB(), shapeOp.getUB()))
      return emitOpError(
          "expected the field type to be larger than the op shape");
    if (!resType.hasEqualShape(shapeOp.getLB(), shapeOp.getUB()))
      return emitOpError("expected op and result type to have the same shape");
  }

  if (!isa<stencil::CastOp>(getField().getDefiningOp()))
    return emitOpError(
        "expected the defining op of the field is a cast operation");

  return success();
}

LogicalResult BufferOp::verify() {
  // Check the temp and result types
  auto tempType = getTemp().getType().cast<stencil::GridType>();
  auto resType = getRes().getType().cast<stencil::GridType>();
  if (resType.getRank() != tempType.getRank())
    return emitOpError("the result and temp types have different rank");
  if (resType.getAllocation() != tempType.getAllocation())
    return emitOpError("the result and temp types have different allocation");
  if (resType.getElementType() != tempType.getElementType())
    return emitOpError(
        "the result and temp types have different element types");

  // Ensure the shape matches the temp and result types
  auto shapeOp = cast<ShapeOp>(this->getOperation());
  if (shapeOp.hasShape()) {
    if (!tempType.hasLargerOrEqualShape(shapeOp.getLB(), shapeOp.getUB()))
      return emitOpError(
          "expected the temp type to be larger than the op shape");
    if (!resType.hasEqualShape(shapeOp.getLB(), shapeOp.getUB()))
      return emitOpError("expected op and result type to have the same shape");
  }

  if (!(isa<stencil::ApplyOp>(getTemp().getDefiningOp()) ||
        isa<stencil::CombineOp>(getTemp().getDefiningOp())))
    return emitOpError("expected buffer to connect to an apply or combine op");

  if (!llvm::all_of(getTemp().getUsers(),
                    [](Operation *op) { return isa<stencil::BufferOp>(op); }))
    return emitOpError("expected only buffers use the same value");
  return success();
}

LogicalResult StoreOp::verify() {
  // Check the field and result types
  auto fieldType = getField().getType().cast<stencil::GridType>();
  auto tempType = getTemp().getType().cast<stencil::GridType>();
  if (fieldType.hasDynamicShape())
    return emitOpError("expected fields to have static shape");
  if (fieldType.getRank() != tempType.getRank())
    return emitOpError("the field and temp types have different rank");
  if (fieldType.getRank() != tempType.getRank())
    return emitOpError("the field and temp types have different rank");
  if (fieldType.getAllocation() != tempType.getAllocation())
    return emitOpError("the field and temp types have different allocation");
  if (fieldType.getElementType() != tempType.getElementType())
    return emitOpError("the field and temp types have different element types");

  // Ensure the shape matches the temp and result types
  auto shapeOp = cast<ShapeOp>(this->getOperation());
  if (!fieldType.hasLargerOrEqualShape(shapeOp.getLB(), shapeOp.getUB()))
    return emitOpError(
        "expected the field type to be larger than the op shape");
  if (!tempType.hasLargerOrEqualShape(shapeOp.getLB(), shapeOp.getUB()))
    return emitOpError("expected the temp type to be larger than the op shape");

  if (!(dyn_cast<stencil::ApplyOp>(getTemp().getDefiningOp()) ||
        dyn_cast<stencil::CombineOp>(getTemp().getDefiningOp())))
    return emitOpError("output temp not result of an apply or a combine op");
  if (llvm::count_if(getField().getUsers(), [](Operation *op) {
        return isa_and_nonnull<stencil::LoadOp>(op);
      }) != 0)
    return emitOpError("an output cannot be an input");
  if (llvm::count_if(getField().getUsers(), [](Operation *op) {
        return isa_and_nonnull<stencil::StoreOp>(op);
      }) != 1)
    return emitOpError("multiple stores to the same output");

  if (!isa<stencil::CastOp>(getField().getDefiningOp()))
    return emitOpError(
        "expected the defining op of the field is a cast operation");

  return success();
}

LogicalResult ApplyOp::verify() {
  // Check the operands
  if (getRegion().front().getNumArguments() != getOperands().size())
    return emitOpError("operand and argument counts do not match");
  for (unsigned i = 0, e = getOperands().size(); i != e; ++i) {
    if (getRegion().front().getArgument(i).getType() !=
        getOperands()[i].getType())
      return emitOpError("operand and argument types do not match");
  }

  // Check the results
  auto shapeOp = cast<ShapeOp>(this->getOperation());
  for (auto result : getRes()) {
    auto tempType = result.getType().cast<GridType>();
    if (shapeOp.hasShape()) {
      if (shapeOp.getRank() != tempType.getRank())
        return emitOpError("expected result rank to match the operation rank");
      if (!tempType.hasEqualShape(shapeOp.getLB(), shapeOp.getUB()))
        return emitOpError("expected temp type to have the shape of the op");
    }
  }
  return success();
}

LogicalResult StoreResultOp::verify() {
  // Check at most one operand
  if (getOperands().size() > 1)
    return emitOpError("expected at most one operand");

  // Check the return type
  auto resultType =
      getRes().getType().cast<stencil::ResultType>().getResultType();
  if (getOperands().size() == 1 && resultType != getOperands()[0].getType())
    return emitOpError("operand type and result type are inconsistent");

  // Check the result mapping
  if (!getReturnOpOperands())
    return emitOpError("expected valid mapping to return op operands");
  return success();
}

LogicalResult ReturnOp::verify() {
  auto applyOp = cast<stencil::ApplyOp>(getOperation()->getParentOp());
  unsigned unrollFac = getUnrollFac();

  // Verify the number of operands matches the number of apply results
  auto results = applyOp.getRes();
  if (getNumOperands() != unrollFac * results.size())
    return emitOpError("the operand and apply result counts do not match");

  // Verify the element types match
  for (unsigned i = 0, e = results.size(); i != e; ++i) {
    auto tempType = applyOp.getResult(i).getType().cast<GridType>();
    for (unsigned j = 0; j < unrollFac; j++)
      if (getOperand(i * unrollFac + j)
              .getType()
              .cast<stencil::ResultType>()
              .getResultType() != tempType.getElementType())
        return emitOpError(
            "the operand and apply result element types do not match");
  }
  return success();
}

ParseResult ApplyOp::parse(OpAsmParser &parser, OperationState &state) {
  SmallVector<OpAsmParser::UnresolvedOperand, 8> operands;
  SmallVector<OpAsmParser::Argument, 8> arguments;
  SmallVector<Type, 8> operandTypes;

  // Parse the assignment list
  if (succeeded(parser.parseOptionalLParen())) {
    do {
      OpAsmParser::Argument currentArgument;
      OpAsmParser::UnresolvedOperand currentOperand;
      Type currentType;

      if (parser.parseArgument(currentArgument, /*allowResultNumber=*/false) || parser.parseEqual() || 
          parser.parseOperand(currentOperand) || parser.parseColonType(currentType))
        return failure();

      currentArgument.type = currentType;
      arguments.push_back(currentArgument);
      operands.push_back(currentOperand);
      operandTypes.push_back(currentType);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }

  // Parse the result types and the optional attributes
  SmallVector<Type, 8> resultTypes;
  if (parser.parseArrowTypeList(resultTypes) ||
      parser.parseOptionalAttrDictWithKeyword(state.attributes))
    return failure();

  // Resolve the operand types
  auto loc = parser.getCurrentLocation();
  if (parser.resolveOperands(operands, operandTypes, loc, state.operands) ||
      parser.addTypesToList(resultTypes, state.types))
    return failure();

  // Parse the body region.
  Region *body = state.addRegion();
  if (parser.parseRegion(*body, arguments))
    return failure();

  // Parse the optional bounds
  ArrayAttr lbAttr, ubAttr;
  if (succeeded(parser.parseOptionalKeyword("to"))) {
    // Parse the optional bounds
    if (parser.parseLParen() ||
        parser.parseAttribute(lbAttr, stencil::ApplyOp::getLBAttrName(),
                              state.attributes) ||
        parser.parseColon() ||
        parser.parseAttribute(ubAttr, stencil::ApplyOp::getUBAttrName(),
                              state.attributes) ||
        parser.parseRParen())
      return failure();
  }

  return success();
}

void ApplyOp::print(OpAsmPrinter &printer) {
  printer << stencil::ApplyOp::getOperationName() << ' ';
  // Print the region arguments
  SmallVector<Value, 10> operands = getOperands();
  if (!getRegion().empty() && !operands.empty()) {
    Block *body = getBody();
    printer << "(";
    llvm::interleaveComma(
        llvm::seq<int>(0, operands.size()), printer, [&](int i) {
          printer << body->getArgument(i) << " = " << operands[i] << " : "
                  << operands[i].getType();
        });
    printer << ") ";
  }

  // Print the result types
  printer << "-> ";
  if (getRes().size() > 1)
    printer << "(";
  llvm::interleaveComma(getRes().getTypes(), printer);
  if (getRes().size() > 1)
    printer << ")";

  // Print optional attributes
  printer.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(), /*elidedAttrs=*/{stencil::ApplyOp::getLBAttrName(),
                                            stencil::ApplyOp::getUBAttrName()});

  // Print region, bounds, and return type
  printer.printRegion(getRegion(),
                      /*printEntryBlockArgs=*/false);
  if (getLb().has_value() && getUb().has_value()) {
    printer << " to (";
    printer.printAttribute(getLb().value());
    printer << " : ";
    printer.printAttribute(getUb().value());
    printer << ")";
  }
}

void stencil::ApplyOp::updateArgumentTypes() {
  for (auto en : llvm::enumerate(getOperandTypes())) {
    if (en.value() != getBody()->getArgument(en.index()).getType()) {
      auto newType = en.value().cast<TempType>();
      auto oldType =
          getBody()->getArgument(en.index()).getType().cast<TempType>();
      // Check both are temporary and only the size changes
      assert(oldType.getElementType() == newType.getElementType() &&
             "expected the same element type");
      assert(oldType.getAllocation() == newType.getAllocation() &&
             "expected the same allocation");
      getBody()->getArgument(en.index()).setType(newType);
    }
  }
}

bool stencil::ApplyOp::hasOnlyEmptyStores() {
  auto result = walk([&](stencil::StoreResultOp resultOp) {
    if (resultOp.getOperands().size() != 0)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return !result.wasInterrupted();
}

ShapeOp stencil::ApplyOp::getCombineTreeRootShape() {
  // Collect all users
  DenseSet<Operation *> users;
  for (auto result : getResults()) {
    for (auto user : result.getUsers()) {
      users.insert(user);
    }
  }

  // Return the shape of the combine tree root if available
  if (users.size() == 1) {
    if (auto combineOp = dyn_cast<CombineOp>(*users.begin())) {
      return cast<ShapeOp>(combineOp.getCombineTreeRoot().getOperation());
    }
  }
  // Otherwise return the shape of the apply operation
  return cast<ShapeOp>(getOperation());
}

//===----------------------------------------------------------------------===//
// stencil.dyn_access
//===----------------------------------------------------------------------===//

void stencil::DynAccessOp::shiftByOffset(ArrayRef<int64_t> offset) {
  // Compute the shifted extent
  Index lb, ub;
  std::tie(lb, ub) = getAccessExtent();
  lb = applyFunElementWise(offset, lb, std::plus<int64_t>());
  ub = applyFunElementWise(offset, ub, std::plus<int64_t>());
  // Create the attributes
  SmallVector<Attribute, kIndexSize> lbAttrs;
  SmallVector<Attribute, kIndexSize> ubAttrs;
  llvm::transform(lb, std::back_inserter(lbAttrs), [&](int64_t x) {
    return IntegerAttr::get(IntegerType::get(getContext(), 64), x);
  });
  llvm::transform(ub, std::back_inserter(ubAttrs), [&](int64_t x) {
    return IntegerAttr::get(IntegerType::get(getContext(), 64), x);
  });
  setLbAttr(ArrayAttr::get(getContext(), lbAttrs));
  setUbAttr(ArrayAttr::get(getContext(), ubAttrs));
}

std::tuple<stencil::Index, stencil::Index>
stencil::DynAccessOp::getAccessExtent() {
  Index lowerBound, upperBound;
  for (auto it : llvm::zip(getLb(), getUb())) {
    lowerBound.push_back(
        std::get<0>(it).cast<IntegerAttr>().getValue().getSExtValue());
    upperBound.push_back(
        std::get<1>(it).cast<IntegerAttr>().getValue().getSExtValue());
  }
  return std::make_tuple(lowerBound, upperBound);
}

//===----------------------------------------------------------------------===//
// stencil.store_result
//===----------------------------------------------------------------------===//

std::optional<SmallVector<OpOperand *, 10>>
stencil::StoreResultOp::getReturnOpOperands() {
  // Keep a list of consumer operands and operations
  DenseSet<Operation *> currOperations;
  SmallVector<OpOperand *, 10> currOperands;
  for (auto &use : getResult().getUses()) {
    currOperands.push_back(&use);
    currOperations.insert(use.getOwner());
  }

  while (currOperations.size() == 1) {
    // Return the results of the return operation
    if (auto returnOp = dyn_cast<stencil::ReturnOp>(*currOperations.begin())) {
      return currOperands;
    }
    // Search the parent block for a return operation
    if (auto yieldOp = dyn_cast<scf::YieldOp>(*currOperations.begin())) {
      // Expected for ops in apply ops not to return a result
      if (isa<scf::ForOp>(yieldOp->getParentOp()) &&
          yieldOp->getParentOfType<stencil::ApplyOp>())
        return {};

      // Search the uses of the result and compute the consumer operations
      currOperations.clear();
      SmallVector<OpOperand *, 10> nextOperands;
      for (auto &use : currOperands) {
        auto result =
            yieldOp->getParentOp()->getResult(use->getOperandNumber());
        for (auto &use : result.getUses()) {
          nextOperands.push_back(&use);
          currOperations.insert(use.getOwner());
        }
      }
      currOperands.swap(nextOperands);
    } else {
      // Expected a return or a yield operation
      return {};
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// stencil.combine
//===----------------------------------------------------------------------===//

namespace {
// Check if operands connect one-by-one to one combine or to multiple apply ops
bool checkOneByOneOperandMapping(OperandRange base, OperandRange extra,
                                 ArrayRef<Operation *> definingOps) {
  // Check the defining op is a unique combine op with one-by-one mapping
  if (auto combineOp = dyn_cast<stencil::CombineOp>(*definingOps.begin())) {
    // Check all operands have one use
    if (!(llvm::all_of(base, [](Value value) { return value.hasOneUse(); }) &&
          llvm::all_of(extra, [](Value value) { return value.hasOneUse(); })))
      return false;
    return definingOps.size() == 1 &&
           combineOp.getNumResults() == base.size() + extra.size();
  }
  // Check the defining ops are apply ops with a one-by-one mapping
  unsigned numResults = 0;
  for (auto definingOp : definingOps) {
    // Check all defining ops are apply ops
    if (!isa<stencil::ApplyOp>(definingOp))
      return false;
    // Check the apply ops connect to combine ops only
    if (llvm::any_of(definingOp->getUsers(), [](Operation *op) {
          return !isa<stencil::CombineOp>(op);
        }))
      return false;
    numResults += definingOp->getNumResults();
  }
  // Check all operands are unique
  DenseSet<Value> operands;
  operands.insert(base.begin(), base.end());
  operands.insert(extra.begin(), extra.end());
  return numResults == operands.size() &&
         numResults == base.size() + extra.size();
}

// Helper to check type compatibility given the combine dim
bool checkTempTypesMatch(Type type1, Type type2, unsigned dim) {
  auto tempType1 = type1.cast<TempType>();
  auto tempType2 = type2.cast<TempType>();
  // Check the element type
  if (tempType1.getElementType() != tempType2.getElementType())
    return false;
  // Check the shape of static shapes match
  for (auto en : llvm::enumerate(tempType1.getShape())) {
    // Skip the combine dim
    if (en.index() == dim)
      continue;
    // Check neither of the sizes is dynamic
    auto size1 = en.value();
    auto size2 = tempType2.getShape()[en.index()];
    if (GridType::isDynamic(size1) || GridType::isDynamic(size2))
      continue;
    // Check the sizes match
    if (size1 != size2)
      return false;
  }
  return true;
}
} // namespace

LogicalResult stencil::CombineOp::verify() {

  auto op = llvm::cast<stencil::CombineOp>(getOperation());

  // Check the combine op has at least one operand
  if (op.getNumOperands() == 0)
    return op.emitOpError("expected the operand list to be non-empty");

  // Check the operand and result sizes match
  if (op.getLower().size() != op.getUpper().size())
    return op.emitOpError("expected the lower and upper operand size to match");
  if (op.getRes().size() !=
      op.getLower().size() + op.getLowerext().size() + op.getUpperext().size())
    return op.emitOpError("expected the result and operand sizes to match");

  // Check all inputs have a defining op
  if (!llvm::all_of(op.getOperands(),
                    [](Value value) { return value.getDefiningOp(); }))
    return op.emitOpError("expected the operands to have a defining op");

  // Check the lower and upper operand types match
  if (!llvm::all_of(
          llvm::zip(op.getLower().getTypes(), op.getUpper().getTypes()),
          [&](std::tuple<Type, Type> x) {
            return checkTempTypesMatch(std::get<0>(x), std::get<1>(x),
                                       op.getDim());
          }))
    return op.emitOpError("expected lower and upper operand types to match");

  // Check the lower/upper operand types match the result types
  if (!llvm::all_of(llvm::zip(op.getLower().getTypes(), op.getRes().getTypes()),
                    [&](std::tuple<Type, Type> x) {
                      return checkTempTypesMatch(std::get<0>(x), std::get<1>(x),
                                                 op.getDim());
                    }))
    return op.emitOpError("expected the lower/upper and result types to match");

  // Check the if the extra types match the corresponding result types
   auto lowerExtResTypes = op.getRes().drop_front(op.getLower().size()).getTypes();
   auto upperExtResTypes = op.getRes().take_back(op.getUpperext().size()).getTypes();
  if (!llvm::all_of(llvm::zip(op.getLowerext().getTypes(), lowerExtResTypes),
                    [&](std::tuple<Type, Type> x) {
                      return checkTempTypesMatch(std::get<0>(x), std::get<1>(x),
                                                 op.getDim());
                    }))
    return op.emitOpError("expected the lowerext and result types to match");
  if (!llvm::all_of(llvm::zip(op.getUpperext().getTypes(), upperExtResTypes),
                    [&](std::tuple<Type, Type> x) {
                      return checkTempTypesMatch(std::get<0>(x), std::get<1>(x),
                                                 op.getDim());
                    }))
    return op.emitOpError("expected the upperext and result types to match");

  // Check the operands either connect to one combine or multiple apply ops
  auto lowerDefiningOps = op.getLowerDefiningOps();
  auto upperDefiningOps = op.getUpperDefiningOps();
  if (!checkOneByOneOperandMapping(op.getLower(), op.getLowerext(),
                                   lowerDefiningOps))
    return op.emitOpError("expected the lower operands to connect one-by-one "
                          "to one combine or multiple apply ops");
  if (!checkOneByOneOperandMapping(op.getUpper(), op.getUpperext(),
                                   upperDefiningOps))
    return op.emitOpError("expected the upper operands to connect one-by-one "
                          "to one combine or multiple apply ops");
  return success();
}

stencil::CombineOp stencil::CombineOp::getCombineTreeRoot() {
  Operation *curr = nullptr;
  Operation *next = this->getOperation();
  do {
    curr = next;
    for (auto user : curr->getUsers()) {
      if (next != curr && next != user) {
        return cast<stencil::CombineOp>(curr);
      }
      next = user;
    }
  } while (isa<stencil::CombineOp>(next));
  return cast<stencil::CombineOp>(curr);
}

//===----------------------------------------------------------------------===//
// Canonicalization
//===----------------------------------------------------------------------===//

stencil::ApplyOpPattern::ApplyOpPattern(MLIRContext *context,
                                        PatternBenefit benefit)
    : OpRewritePattern<stencil::ApplyOp>(context, benefit) {}

stencil::ApplyOp
stencil::ApplyOpPattern::cleanupOpArguments(stencil::ApplyOp applyOp,
                                            PatternRewriter &rewriter) const {
  // Compute the new operand list and index mapping
  llvm::DenseMap<Value, unsigned int> newIndex;
  SmallVector<Value, 10> newOperands;
  for (auto en : llvm::enumerate(applyOp.getOperands())) {
    if (newIndex.count(en.value()) == 0) {
      if (!applyOp.getBody()->getArgument(en.index()).getUses().empty()) {
        newIndex[en.value()] = newOperands.size();
        newOperands.push_back(en.value());
      }
    }
  }

  // Create a new operation with shorther argument list
  if (newOperands.size() < applyOp.getNumOperands()) {
    auto loc = applyOp.getLoc();
    auto newOp = rewriter.create<stencil::ApplyOp>(
        loc, applyOp.getResultTypes(), newOperands, applyOp.getLb(),
        applyOp.getUb());

    // Compute the argument mapping and move the block
    SmallVector<Value, 10> newArgs(applyOp.getNumOperands());
    llvm::transform(applyOp.getOperands(), newArgs.begin(), [&](Value value) {
      return newIndex.count(value) == 0
                 ? nullptr // pass default value if the new apply has no params
                 : newOp.getBody()->getArgument(newIndex[value]);
    });
    rewriter.mergeBlocks(applyOp.getBody(), newOp.getBody(), newArgs);
    return newOp;
  }
  return nullptr;
}

stencil::CombineOpPattern::CombineOpPattern(MLIRContext *context,
                                            PatternBenefit benefit)
    : OpRewritePattern<stencil::CombineOp>(context, benefit) {}

stencil::ApplyOp stencil::CombineOpPattern::createEmptyApply(
    stencil::CombineOp combineOp, int64_t lowerLimit, int64_t upperLimit,
    ValueRange values, PatternRewriter &rewriter) const {
  // Get the location of the mirrored return operation
  auto loc = combineOp.getLoc();

  // Get the return op attached to the operand range
  auto applyOp = cast<stencil::ApplyOp>(values.front().getDefiningOp());
  auto returnOp = cast<stencil::ReturnOp>(applyOp.getBody()->getTerminator());

  // Get the shape of the combine op
  auto shapeOp = cast<ShapeOp>(combineOp.getOperation());
  Index lb, ub;

  // Compute the result types depending on the size information
  SmallVector<Type, 10> newResultTypes;
  if (shapeOp.hasShape()) {
    // Compute the shape of the empty apply
    lb = shapeOp.getLB();
    ub = shapeOp.getUB();
    lb[combineOp.getDim()] = max(lowerLimit, lb[combineOp.getDim()]);
    ub[combineOp.getDim()] = min(upperLimit, ub[combineOp.getDim()]);

    // Resize the operand types
    for (auto value : values) {
      auto operandType = value.getType().cast<TempType>();
      auto shape = applyFunElementWise(ub, lb, std::minus<int64_t>());
      newResultTypes.push_back(
          TempType::get(operandType.getElementType(), shape));
    }
  } else {
    // Assume the types have a dynamic shape
    for (auto value : values) {
      auto operandType = value.getType().cast<TempType>();
      assert(operandType.hasDynamicShape() &&
             "expected operand type to have a dynamic shape");
    }
  }

  // Create an empty apply op including empty stores
  auto newOp = rewriter.create<stencil::ApplyOp>(
      returnOp.getLoc(), newResultTypes,
      lb.empty() ? nullptr : rewriter.getI64ArrayAttr(lb),
      ub.empty() ? nullptr : rewriter.getI64ArrayAttr(ub));
  newOp.getRegion().push_back(new Block());

  // Update the body of the apply op
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(newOp.getBody());

  // Create the empty stores and the return op
  SmallVector<Value, 10> newOperands;
  for (auto newResultType : newResultTypes) {
    auto elementType = newResultType.cast<TempType>().getElementType();
    auto resultOp = rewriter.create<stencil::StoreResultOp>(
        loc, ResultType::get(elementType), ValueRange());
    newOperands.append(returnOp.getUnrollFac(), resultOp);
  }
  rewriter.create<stencil::ReturnOp>(loc, newOperands, returnOp.getUnroll());
  return newOp;
}

namespace {

/// This is a pattern to remove duplicate loads
struct ApplyOpLoadCleaner : public stencil::ApplyOpPattern {
  using ApplyOpPattern::ApplyOpPattern;

  LogicalResult cleanupLoadOps(DenseSet<Operation *> &loadOps,
                               stencil::ApplyOp applyOp,
                               PatternRewriter &rewriter) const {
    // Check all load ops have a shape (otherwise cse is sufficient)
    if (llvm::any_of(loadOps,
                     [](Operation *op) { return !llvm::cast<ShapeOp>(op).hasShape(); }))
      return failure();

    // Compute the bounding box of all load shapes
    auto lb = cast<ShapeOp>(*loadOps.begin()).getLB();
    auto ub = cast<ShapeOp>(*loadOps.begin()).getUB();
    for (auto loadOp : loadOps) {
      auto shapeOp = cast<ShapeOp>(loadOp);
      lb = applyFunElementWise(shapeOp.getLB(), lb, min);
      ub = applyFunElementWise(shapeOp.getUB(), ub, max);
    }

    // Create a new load operation
    auto loadOp = rewriter.create<stencil::LoadOp>(
        applyOp.getLoc(), cast<stencil::LoadOp>(*loadOps.begin()).getField(),
        rewriter.getI64ArrayAttr(lb), rewriter.getI64ArrayAttr(ub));

    // Compute the new operand list
    SmallVector<Value, 10> newOperands;
    llvm::transform(applyOp.getOperands(), std::back_inserter(newOperands),
                    [&](Value value) {
                      return loadOps.count(value.getDefiningOp()) == 1 ? loadOp
                                                                       : value;
                    });

    // Replace the apply operation using the new load op
    auto newOp = rewriter.create<stencil::ApplyOp>(
        applyOp.getLoc(), applyOp.getResultTypes(), newOperands,
        applyOp.getLb(), applyOp.getUb());
    rewriter.mergeBlocks(applyOp.getBody(), newOp.getBody(),
                         newOp.getBody()->getArguments());
    rewriter.replaceOp(applyOp, newOp.getResults());
    return success();
  }

  LogicalResult matchAndRewrite(stencil::ApplyOp applyOp,
                                PatternRewriter &rewriter) const override {
    // Compute mapping of the loaded fields to the load ops
    DenseMap<Value, DenseSet<Operation *>> fieldToLoadOps;
    for (auto value : applyOp.getOperands()) {
      if (auto loadOp =
              dyn_cast_or_null<stencil::LoadOp>(value.getDefiningOp())) {
        fieldToLoadOps[loadOp.getField()].insert(loadOp.getOperation());
      }
    }
    // Replace multiple loads of the same field
    for (auto entry : fieldToLoadOps) {
      if (entry.getSecond().size() > 1) {
        return cleanupLoadOps(entry.getSecond(), applyOp, rewriter);
      }
    }
    return failure();
  }
};

/// This is a pattern to remove duplicate and unused arguments
struct ApplyOpArgCleaner : public stencil::ApplyOpPattern {
  using ApplyOpPattern::ApplyOpPattern;

  LogicalResult matchAndRewrite(stencil::ApplyOp applyOp,
                                PatternRewriter &rewriter) const override {
    if (auto newOp = cleanupOpArguments(applyOp, rewriter)) {
      rewriter.replaceOp(applyOp, newOp.getResults());
      return success();
    }
    return failure();
  }
};

/// This is a pattern removes unused results
struct ApplyOpResCleaner : public stencil::ApplyOpPattern {
  using ApplyOpPattern::ApplyOpPattern;

  LogicalResult matchAndRewrite(stencil::ApplyOp applyOp,
                                PatternRewriter &rewriter) const override {
    // Compute the updated result list
    SmallVector<OpResult, 10> usedResults;
    llvm::copy_if(applyOp.getResults(), std::back_inserter(usedResults),
                  [](OpResult result) { return !result.use_empty(); });

    if (usedResults.size() != applyOp.getNumResults()) {
      // Erase the op if it has not uses
      if (usedResults.size() == 0) {
        rewriter.eraseOp(applyOp);
        return success();
      }

      // Get the return operation
      auto returnOp =
          cast<stencil::ReturnOp>(applyOp.getBody()->getTerminator());
      unsigned unrollFac = returnOp.getUnrollFac();

      // Compute the new result and and return op operand vector
      SmallVector<Type, 10> newResultTypes;
      SmallVector<Value, 10> newOperands;
      for (auto usedResult : usedResults) {
        newResultTypes.push_back(usedResult.getType());
        auto slice = returnOp.getOperands().slice(
            usedResult.getResultNumber() * unrollFac, unrollFac);
        newOperands.append(slice.begin(), slice.end());
      }

      // Create a new apply operation
      auto newOp = rewriter.create<stencil::ApplyOp>(
          applyOp.getLoc(), newResultTypes, applyOp.getOperands(),
          applyOp.getLb(), applyOp.getUb());
      rewriter.setInsertionPoint(returnOp);
      rewriter.create<stencil::ReturnOp>(returnOp.getLoc(), newOperands,
                                         returnOp.getUnroll());
      rewriter.eraseOp(returnOp);
      rewriter.mergeBlocks(applyOp.getBody(), newOp.getBody(),
                           newOp.getBody()->getArguments());

      // Compute the replacement results
      SmallVector<Value, 10> repResults(applyOp.getNumResults(),
                                        newOp.getResults().front());
      for (auto en : llvm::enumerate(usedResults))
        repResults[en.value().getResultNumber()] = newOp.getResult(en.index());
      rewriter.replaceOp(applyOp, repResults);
      return success();
    }
    return failure();
  }
};

/// This is a pattern to removes combines with symmetric operands
struct CombineOpSymmetricCleaner : public stencil::CombineOpPattern {
  using CombineOpPattern::CombineOpPattern;

  stencil::ApplyOp getDefiningApplyOp(Value value) const {
    return dyn_cast_or_null<stencil::ApplyOp>(value.getDefiningOp());
  }

  LogicalResult matchAndRewrite(stencil::CombineOp combineOp,
                                PatternRewriter &rewriter) const override {
    // Exit if the combine has extra operands
    if (combineOp.getLowerext().size() > 0 ||
        combineOp.getUpperext().size() > 0)
      return failure();

    // Compute the empty values
    SmallVector<Value, 10> emptyValues;
    for (auto en : llvm::enumerate(combineOp.getLower())) {
      auto lowerOp = getDefiningApplyOp(en.value());
      auto upperOp = getDefiningApplyOp(combineOp.getUpper()[en.index()]);
      if (lowerOp && upperOp && lowerOp.hasOnlyEmptyStores() &&
          upperOp.hasOnlyEmptyStores()) {
        emptyValues.push_back(en.value());
      }
    }

    // Compare the upper and lower values
    for (auto en : llvm::enumerate(combineOp.getLower())) {
      if (en.value() != combineOp.getUpper()[en.index()] &&
          !llvm::is_contained(emptyValues, en.value()))
        return failure();
    }

    // Create an empty apply
    ApplyOp emptyOp;
    if (!emptyValues.empty()) {
      emptyOp = createEmptyApply(combineOp, std::numeric_limits<int64_t>::min(),
                                 std::numeric_limits<int64_t>::max(),
                                 emptyValues, rewriter);
    }

    // Compute the replacement values
    unsigned emptyCount = 0;
    SmallVector<Value, 10> repResults;
    for (auto en : llvm::enumerate(combineOp.getLower())) {
      if (en.value() == combineOp.getUpper()[en.index()]) {
        repResults.push_back(en.value());
      } else {
        repResults.push_back(emptyOp.getResult(emptyCount++));
      }
    }

    // Replace the combine op
    rewriter.replaceOp(combineOp, repResults);
    return success();
  }
};

/// This is a pattern to remove combines that do not split the domain
struct CombineOpEmptyCleaner : public stencil::CombineOpPattern {
  using CombineOpPattern::CombineOpPattern;

  LogicalResult matchAndRewrite(stencil::CombineOp combineOp,
                                PatternRewriter &rewriter) const override {
    // Check if the index of the combine op is inside the shape
    auto shapeOp = cast<ShapeOp>(combineOp.getOperation());
    if (shapeOp.hasShape()) {
      // Remove the upper operands if the index is larger than the upper bound
      if (combineOp.getMyIndex() > shapeOp.getUB()[combineOp.getDim()]) {
        // Compute the replacement results
        SmallVector<Value, 10> repResults = combineOp.getLower();
        repResults.append(combineOp.getLowerext().begin(),
                          combineOp.getLowerext().end());

        // Introduce empty stores in case there are upper extra results
        if (combineOp.getUpperext().size() > 0) {
          auto newOp =
              createEmptyApply(combineOp, std::numeric_limits<int64_t>::min(),
                               std::numeric_limits<int64_t>::max(),
                               combineOp.getUpperext(), rewriter);
          repResults.append(newOp.getResults().begin(),
                            newOp.getResults().end());
        }

        // Replace the combine op
        rewriter.replaceOp(combineOp, repResults);
        return success();
      }
      // Remove the lower operands if the index is smaller than the lower bound
      if (combineOp.getMyIndex() < shapeOp.getLB()[combineOp.getDim()]) {
        // Compute the replacement results
        SmallVector<Value, 10> repResults = combineOp.getUpper();

        // Introduce empty stores in case there are lower extra results
        if (combineOp.getLowerext().size() > 0) {
          auto newOp =
              createEmptyApply(combineOp, std::numeric_limits<int64_t>::min(),
                               std::numeric_limits<int64_t>::max(),
                               combineOp.getLowerext(), rewriter);
          repResults.append(newOp.getResults().begin(),
                            newOp.getResults().end());
        }
        repResults.append(combineOp.getUpperext().begin(),
                          combineOp.getUpperext().end());

        // Replace the combine op
        rewriter.replaceOp(combineOp, repResults);
        return success();
      }
    }
    return failure();
  }
};

/// This is a pattern to remove unused arguments
struct CombineOpResCleaner : public stencil::CombineOpPattern {
  using CombineOpPattern::CombineOpPattern;

  LogicalResult matchAndRewrite(stencil::CombineOp combineOp,
                                PatternRewriter &rewriter) const override {
    // Compute the updated result list
    SmallVector<OpResult, 10> usedResults;
    llvm::copy_if(combineOp.getResults(), std::back_inserter(usedResults),
                  [](OpResult result) { return !result.use_empty(); });

    if (usedResults.size() != combineOp.getNumResults()) {
      // Erase the op if it has not uses
      if (usedResults.size() == 0) {
        rewriter.eraseOp(combineOp);
        return success();
      }

      // Compute the new result types and operands
      SmallVector<Type, 10> newResultTypes;
      llvm::transform(usedResults, std::back_inserter(newResultTypes),
                      [](Value value) { return value.getType(); });

      SmallVector<Value, 10> newLowerOperands, newLowerExtraOperands;
      SmallVector<Value, 10> newUpperOperands, newUpperExtraOperands;
      for (auto used : usedResults) {
        unsigned resultNumber = used.getResultNumber();
        // Copy the main operands
        if (auto num = combineOp.getLowerOperandNumber(resultNumber)) {
          newLowerOperands.push_back(combineOp.getLower()[num.value()]);
          newUpperOperands.push_back(combineOp.getUpper()[num.value()]);
        }
        // Copy the lower extra operands
        if (auto num = combineOp.getLowerExtraOperandNumber(resultNumber)) {
          newLowerExtraOperands.push_back(combineOp.getLowerext()[num.value()]);
        }
        // Copy the upper extra operands
        if (auto num = combineOp.getUpperExtraOperandNumber(resultNumber)) {
          newUpperExtraOperands.push_back(combineOp.getUpperext()[num.value()]);
        }
      }

      // Create a new combine op that returns only the used results
      auto newOp = rewriter.create<stencil::CombineOp>(
          combineOp.getLoc(), newResultTypes, combineOp.getDim(),
          combineOp.getMyIndex(), newLowerOperands, newUpperOperands,
          newLowerExtraOperands, newUpperExtraOperands, combineOp.getLbAttr(),
          combineOp.getUbAttr());

      // Compute the replacement results
      SmallVector<Value, 10> repResults(combineOp.getNumResults(),
                                        newOp.getResults().front());
      for (auto en : llvm::enumerate(usedResults))
        repResults[en.value().getResultNumber()] = newOp.getResult(en.index());
      rewriter.replaceOp(combineOp, repResults);
      return success();
    }
    return failure();
  }
};

// Helper methods to hoist operations
LogicalResult hoistBackward(Operation *op, PatternRewriter &rewriter,
                            std::function<bool(Operation *)> condition) {
  // Skip compute operations
  auto curr = op;
  while (curr->getPrevNode() && condition(curr->getPrevNode()) &&
         !llvm::is_contained(curr->getPrevNode()->getUsers(), op))
    curr = curr->getPrevNode();

  // Move the operation
  if (curr != op) {
    rewriter.setInsertionPoint(curr);
    rewriter.replaceOp(op, rewriter.clone(*op)->getResults());
    return success();
  }
  return failure();
}
LogicalResult hoistForward(Operation *op, PatternRewriter &rewriter,
                           std::function<bool(Operation *)> condition) {
  // Skip compute operations
  auto curr = op;
  while (curr->getNextNode() && condition(curr->getNextNode()) &&
         !curr->getNextNode()->hasTrait<OpTrait::IsTerminator>())
    curr = curr->getNextNode();

  // Move the operation
  if (curr != op) {
    rewriter.setInsertionPointAfter(curr);
    rewriter.replaceOp(op, rewriter.clone(*op)->getResults());
    return success();
  }
  return failure();
} // namespace

/// This is a pattern to hoist assert ops out of the computation
struct CastOpHoisting : public OpRewritePattern<stencil::CastOp> {
  using OpRewritePattern<stencil::CastOp>::OpRewritePattern;

  // Remove duplicates if needed
  LogicalResult matchAndRewrite(stencil::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    // Skip all operations except for other casts
    auto condition = [](Operation *op) { return !isa<stencil::CastOp>(op); };
    return hoistBackward(castOp.getOperation(), rewriter, condition);
  }
};

/// This is a pattern to hoist load ops out of the computation
struct LoadOpHoisting : public OpRewritePattern<stencil::LoadOp> {
  using OpRewritePattern<stencil::LoadOp>::OpRewritePattern;

  // Remove duplicates if needed
  LogicalResult matchAndRewrite(stencil::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    // Skip all operations except for casts and other loads
    auto condition = [](Operation *op) {
      return !isa<stencil::LoadOp>(op) && !isa<stencil::CastOp>(op);
    };
    return hoistBackward(loadOp.getOperation(), rewriter, condition);
  }
};

/// This is a pattern to hoist store ops out of the computation
struct StoreOpHoisting : public OpRewritePattern<stencil::StoreOp> {
  using OpRewritePattern<stencil::StoreOp>::OpRewritePattern;

  // Remove duplicates if needed
  LogicalResult matchAndRewrite(stencil::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    // Skip all operations except for stores
    auto condition = [](Operation *op) { return !isa<stencil::StoreOp>(op); };
    return hoistForward(storeOp.getOperation(), rewriter, condition);
  }
};

} // end anonymous namespace

// Register canonicalization patterns
void stencil::ApplyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.insert<ApplyOpArgCleaner, ApplyOpResCleaner, ApplyOpLoadCleaner>(
      context);
}

void stencil::CombineOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results.insert<CombineOpResCleaner, CombineOpEmptyCleaner,
                 CombineOpSymmetricCleaner>(context);
}

void stencil::CastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.insert<CastOpHoisting>(context);
}
void stencil::LoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.insert<LoadOpHoisting>(context);
}
void stencil::StoreOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.insert<StoreOpHoisting>(context);
}

namespace mlir {
namespace stencil {
#include "Dialect/Stencil/StencilOpsInterfaces.cpp.inc"
} // namespace stencil
} // namespace mlir

#define GET_OP_CLASSES
#include "Dialect/Stencil/StencilOps.cpp.inc"
