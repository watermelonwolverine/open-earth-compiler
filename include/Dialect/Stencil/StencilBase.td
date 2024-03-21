#ifndef Stencil_BASE
#define Stencil_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Stencil Dialect
//===----------------------------------------------------------------------===//

def Stencil_Dialect : Dialect {
    let name = "stencil";
    let description = [{A stencil dialect for weather and climate.}];
    let cppNamespace = "::mlir::stencil";
}

//===----------------------------------------------------------------------===//
// Stencil Types
//===----------------------------------------------------------------------===//

def Stencil_IsTempType : CPred<"$_self.isa<::mlir::stencil::TempType>()">;
def Stencil_IsFieldType : CPred<"$_self.isa<::mlir::stencil::FieldType>()">;
def Stencil_IsResultType : CPred<"$_self.isa<::mlir::stencil::ResultType>()">;

def Stencil_Temp : Type<Stencil_IsTempType, "a temporary value">;
def Stencil_Field : Type<Stencil_IsFieldType, "a input/output array">;
def Stencil_Result : Type<Stencil_IsResultType, "a result">;

def Stencil_Element : AnyTypeOf<[F32, F64]>;

def Stencil_Index : ConfinedAttr<I64ArrayAttr, [ArrayCount<3>]>;
def Stencil_Loop : ConfinedAttr<I64ArrayAttr, [ArrayCount<4>]>;

//===----------------------------------------------------------------------===//
// Stencil Operation
//===----------------------------------------------------------------------===//

// Base class for all Stencil ops.
class Stencil_Op<string mnemonic, list<Trait> traits = []> :
    Op<Stencil_Dialect, mnemonic, traits>{
}

#endif // Stencil_BASE
