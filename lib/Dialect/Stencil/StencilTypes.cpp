#include "Dialect/Stencil/StencilTypes.h"
#include "Dialect/Stencil/StencilDialect.h"
#include "Dialect/Stencil/StencilUtils.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>

using namespace mlir;
using namespace stencil;

//===----------------------------------------------------------------------===//
// GridType
//===----------------------------------------------------------------------===//

constexpr int64_t GridType::kDynamicDimension;
constexpr int64_t GridType::kScalarDimension;

bool GridType::classof(Type type) { return type.isa<FieldType, TempType>(); }

Type GridType::getElementType() const {
  return static_cast<ImplType *>(impl)->getElementType();
}

ArrayRef<int64_t> GridType::getShape() const {
  return static_cast<ImplType *>(impl)->getShape();
}

unsigned GridType::getRank() const { return (unsigned)getShape().size(); }

int64_t GridType::hasDynamicShape() const {
  return llvm::all_of(getShape(), [](int64_t size) {
    return size == kDynamicDimension || size == kScalarDimension;
  });
}

int64_t GridType::hasStaticShape() const {
  return llvm::none_of(getShape(),
                       [](int64_t size) { return size == kDynamicDimension; });
}

bool GridType::hasEqualShape(ArrayRef<int64_t> lb,
                                ArrayRef<int64_t> ub) const {
  auto shape = applyFunElementWise(ub, lb, std::minus<int64_t>());
  return llvm::all_of(llvm::zip(getAllocation(), getShape(), shape),
                      [&](std::tuple<bool, int64_t, int64_t> x) {
                        return !std::get<0>(x) ||
                               GridType::isDynamic(std::get<1>(x)) ||
                               (std::get<1>(x) == std::get<2>(x));
                      });
}

bool GridType::hasLargerOrEqualShape(ArrayRef<int64_t> lb,
                                        ArrayRef<int64_t> ub) const {
  auto shape = applyFunElementWise(ub, lb, std::minus<int64_t>());
  return llvm::all_of(llvm::zip(getAllocation(), getShape(), shape),
                      [&](std::tuple<bool, int64_t, int64_t> x) {
                        return !std::get<0>(x) ||
                               GridType::isDynamic(std::get<1>(x)) ||
                               (std::get<1>(x) >= std::get<2>(x));
                      });
}

SmallVector<bool, 3> GridType::getAllocation() const {
  SmallVector<bool, 3> result;
  result.resize(getRank());
  llvm::transform(getShape(), result.begin(),
                  [](int64_t x) { return x != kScalarDimension; });
  return result;
}

SmallVector<int64_t, 3> GridType::getMemRefShape() const {
  SmallVector<int64_t, 3> result;
  for (auto size : llvm::reverse(getShape())) {
    switch (size) {
    case (kDynamicDimension):
      result.push_back(ShapedType::kDynamic);
      break;
    case (kScalarDimension):
      break;
    default:
      result.push_back(size);
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// FieldType
//===----------------------------------------------------------------------===//

FieldType FieldType::get(Type elementType, llvm::ArrayRef<int64_t> shape) {
  return Base::get(elementType.getContext(), elementType, shape);
}

//===----------------------------------------------------------------------===//
// TempType
//===----------------------------------------------------------------------===//

TempType TempType::get(Type elementType, llvm::ArrayRef<int64_t> shape) {
  return Base::get(elementType.getContext(), elementType, shape);
}

TempType TempType::get(Type elementType, ArrayRef<bool> allocation,
                       ArrayRef<int64_t> lb, ArrayRef<int64_t> ub) {
  auto shape = applyFunElementWise(ub, lb, std::minus<int64_t>());
  for (auto en : llvm::enumerate(allocation)) {
    if (!en.value())
      shape[en.index()] = GridType::kScalarDimension;
  }
  return TempType::get(elementType, shape);
}

TempType TempType::get(Type elementType, ArrayRef<bool> allocation) {
  SmallVector<int64_t, kIndexSize> shape;
  for (auto hasAllocation : allocation) {
    shape.push_back(hasAllocation ? GridType::kDynamicDimension
                                  : GridType::kScalarDimension);
  }
  return TempType::get(elementType, shape);
}

//===----------------------------------------------------------------------===//
// ResultType
//===----------------------------------------------------------------------===//

ResultType ResultType::get(Type resultType) {
  return Base::get(resultType.getContext(), resultType);
}

Type ResultType::getResultType() const { return getImpl()->getResultType(); }