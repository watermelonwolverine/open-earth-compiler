#ifndef DIALECT_STENCIL_STENCILTYPES_H
#define DIALECT_STENCIL_STENCILTYPES_H

#include "Dialect/Stencil/StencilDialect.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>

namespace mlir {
namespace stencil {

namespace detail {
struct GridTypeStorage : public TypeStorage {
  GridTypeStorage(Type elementTy, size_t size, const int64_t *shape)
      : TypeStorage(), elementType(elementTy), size(size), shape(shape) {}

  /// Hash key used for uniquing
  using KeyTy = std::pair<Type, ArrayRef<int64_t>>;

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, getShape());
  }

  Type getElementType() const { return elementType; }
  ArrayRef<int64_t> getShape() const { return {shape, size}; }

  Type elementType;
  const size_t size;
  const int64_t *shape;
};

struct FieldTypeStorage : public GridTypeStorage {
  using GridTypeStorage::GridTypeStorage;

  /// Construction
  static FieldTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    // Copy the allocation into the bump pointer.
    ArrayRef<int64_t> shape = allocator.copyInto(key.second);

    return new (allocator.allocate<FieldTypeStorage>())
        FieldTypeStorage(key.first, shape.size(), shape.data());
  }
};

struct TempTypeStorage : public GridTypeStorage {
  using GridTypeStorage::GridTypeStorage;

  /// Construction
  static TempTypeStorage *construct(TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    // Copy the allocation into the bump pointer.
    ArrayRef<int64_t> shape = allocator.copyInto(key.second);

    return new (allocator.allocate<TempTypeStorage>())
        TempTypeStorage(key.first, shape.size(), shape.data());
  }
};
struct ResultTypeStorage : public TypeStorage {
  ResultTypeStorage(Type resultType) : TypeStorage(), resultType(resultType) {}

  /// Hash key used for uniquing
  using KeyTy = Type;

  bool operator==(const KeyTy &key) const { return key == resultType; }

  Type getResultType() const { return resultType; }

  /// Construction
  static ResultTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<ResultTypeStorage>()) ResultTypeStorage(key);
  }

  Type resultType;
};
} // namespace detail

//===----------------------------------------------------------------------===//
// GridType
//===----------------------------------------------------------------------===//

/// Base class of the field and view types.
class GridType : public Type {
public:
  using ImplType = detail::GridTypeStorage;
  using Type::Type;

  static bool classof(Type type);

  /// Constants used to mark dynamic size or scalarized dimensions
  static constexpr int64_t kDynamicDimension = ShapedType::kDynamic;
  static constexpr int64_t kScalarDimension = 0;

  /// Return the element type
  Type getElementType() const;

  /// Return the shape of the type
  ArrayRef<int64_t> getShape() const;

  /// Return the rank of the type
  unsigned getRank() const;

  /// Return true if all dimensions have a dynamic shape
  int64_t hasDynamicShape() const;

  /// Return true if all dimensions have a static
  int64_t hasStaticShape() const;

  /// Return true if the type matches the shape
  bool hasEqualShape(ArrayRef<int64_t> lb, ArrayRef<int64_t> ub) const;

  /// Return true if the type is larger or equal than the shape
  bool hasLargerOrEqualShape(ArrayRef<int64_t> lb, ArrayRef<int64_t> ub) const;

  /// Return the allocated / non-scalar dimensions
  SmallVector<bool, 3> getAllocation() const;

  /// Return the compatible memref shape
  /// (reverse shape from column-major to row-major)
  SmallVector<int64_t, 3> getMemRefShape() const;

  /// Return true if the dimension size is dynamic
  static constexpr bool isDynamic(int64_t dimSize) {
    return dimSize == kDynamicDimension;
  }

  /// Return true for scalarized dimensions
  static constexpr bool isScalar(int64_t dimSize) {
    return dimSize == kScalarDimension;
  }
}; // namespace stencil

//===----------------------------------------------------------------------===//
// FieldType
//===----------------------------------------------------------------------===//

/// Fields are multi-dimensional input and output arrays
class FieldType
    : public Type::TypeBase<FieldType, GridType, detail::FieldTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "stencil.field";
  static FieldType get(Type elementType, ArrayRef<int64_t> shape);
};

//===----------------------------------------------------------------------===//
// TempType
//===----------------------------------------------------------------------===//

/// Temporaries keep multi-dimensional intermediate results
class TempType
    : public Type::TypeBase<TempType, GridType, detail::TempTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "stencil.temp";
  static TempType get(Type elementType, ArrayRef<int64_t> shape);

  /// Get a statically sized temp type
  static TempType get(Type elementType, ArrayRef<bool> allocation,
                      ArrayRef<int64_t> lb, ArrayRef<int64_t> ub);

  /// Get a dynamically sized temp type
  static TempType get(Type elementType, ArrayRef<bool> allocation);
};

//===----------------------------------------------------------------------===//
// ResultType
//===----------------------------------------------------------------------===//

/// Temporaries keep multi-dimensional intermediate results
class ResultType
    : public Type::TypeBase<ResultType, Type, detail::ResultTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "stencil.result";
  static ResultType get(Type resultType);

  /// Return the result type
  Type getResultType() const;
};

} // namespace stencil
} // namespace mlir

#endif // DIALECT_STENCIL_STENCILTYPES_H
