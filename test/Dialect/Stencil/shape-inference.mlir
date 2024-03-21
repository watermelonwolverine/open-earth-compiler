// RUN: oec-opt %s -split-input-file --stencil-shape-inference | oec-opt | FileCheck %s
// RUN: oec-opt %s -split-input-file --stencil-shape-inference='extend-storage' | oec-opt | FileCheck --check-prefix=CHECKEXT %s

// CHECK-LABEL: func @simple(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
func @simple(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  //  CHECK: %{{.*}} = stencil.load %{{.*}}([-1, -2, 0] : [65, 66, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<66x68x60xf64>
  %2 = stencil.load %0 : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
  //  CHECK: %{{.*}} = stencil.apply (%{{.*}} = %{{.*}} : !stencil.temp<66x68x60xf64>) -> !stencil.temp<64x64x60xf64> {
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %4 = stencil.access %arg2 [-1, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = stencil.access %arg2 [1, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %6 = arith.addf %4, %5 : f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  //  CHECK: } to ([0, 0, 0] : [64, 64, 60])
  }
  //  CHECK: stencil.store %{{.*}} to %{{.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %3 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @multiple(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
func @multiple(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  //  CHECK: %{{.*}} = stencil.load %{{.*}}([-1, -2, 0] : [65, 66, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<66x68x60xf64>
  %2 = stencil.load %0 : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
  //  CHECK: %{{.*}} = stencil.apply (%{{.*}} = %{{.*}} : !stencil.temp<66x68x60xf64>) -> !stencil.temp<66x64x60xf64> {
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %5 = stencil.access %arg2 [0, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %6 = stencil.access %arg2 [0, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %7 = arith.addf %5, %6 : f64
    %8 = stencil.store_result %7 : (f64) -> !stencil.result<f64>
    stencil.return %8 : !stencil.result<f64>
  //  CHECK: } to ([-1, 0, 0] : [65, 64, 60])
  }
  //  CHECK: %{{.*}} = stencil.apply (%{{.*}} = %{{.*}} : !stencil.temp<66x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
  %4 = stencil.apply (%arg2 = %3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %9 = stencil.access %arg2 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %10 = stencil.access %arg2 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %11 = arith.addf %9, %10 : f64
    %12 = stencil.store_result %11 : (f64) -> !stencil.result<f64>
    stencil.return %12 : !stencil.result<f64>
  //  CHECK: } to ([0, 0, 0] : [64, 64, 60])
  }
  //  CHECK: stencil.store %{{.*}} to %{{.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %4 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @lower(%{{.*}}: !stencil.field<?x?x0xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
func @lower(%arg0: !stencil.field<?x?x0xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x0xf64>) -> !stencil.field<70x70x0xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  //  CHECK: %{{.*}} = stencil.load %{{.*}}([-1, -2, 0] : [65, 66, 60]) : (!stencil.field<70x70x0xf64>) -> !stencil.temp<66x68x0xf64>
  %2 = stencil.load %0 : (!stencil.field<70x70x0xf64>) -> !stencil.temp<?x?x0xf64>
  //  CHECK: %{{.*}} = stencil.apply (%{{.*}} = %{{.*}} : !stencil.temp<66x68x0xf64>) -> !stencil.temp<64x64x60xf64> {
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<?x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
    %4 = stencil.access %arg2 [-1, 2, 0] : (!stencil.temp<?x?x0xf64>) -> f64
    %5 = stencil.access %arg2 [1, -2, 0] : (!stencil.temp<?x?x0xf64>) -> f64
    %6 = arith.addf %4, %5 : f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  //  CHECK: } to ([0, 0, 0] : [64, 64, 60])
  }
  //  CHECK: stencil.store %{{.*}} to %{{.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %3 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @twostores(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
// CHECKEXT-LABEL: func @twostores(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
func @twostores(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  // CHECK: %{{.*}}:2 = stencil.apply -> (!stencil.temp<64x66x60xf64>, !stencil.temp<64x66x60xf64>) {
  // CHECKEXT: %{{.*}}:2 = stencil.apply -> (!stencil.temp<64x66x60xf64>, !stencil.temp<64x66x60xf64>) {
  %2,%3 = stencil.apply -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
    %4 = arith.constant 1.0 : f64
    %5 = stencil.store_result %4 : (f64) -> !stencil.result<f64>
    %6 = stencil.store_result %4 : (f64) -> !stencil.result<f64>
    stencil.return %5, %6 : !stencil.result<f64>, !stencil.result<f64>
  // CHECK: } to ([0, -1, 0] : [64, 65, 60])
  // CHECKEXT: } to ([0, -1, 0] : [64, 65, 60])
  }
  // CHECK: stencil.store %{{.*}} to %{{.*}}([0, 0, 0] : [64, 65, 60]) : !stencil.temp<64x66x60xf64> to !stencil.field<70x70x60xf64>
  // CHECKEXT: stencil.store %{{.*}} to %{{.*}}([0, -1, 0] : [64, 65, 60]) : !stencil.temp<64x66x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %2 to %0([0, 0, 0] : [64, 65, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  // CHECK: stencil.store %{{.*}} to %{{.*}}([0, -1, 0] : [64, 64, 60]) : !stencil.temp<64x66x60xf64> to !stencil.field<70x70x60xf64>
  // CHECKEXT: stencil.store %{{.*}} to %{{.*}}([0, -1, 0] : [64, 65, 60]) : !stencil.temp<64x66x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %3 to %1([0, -1, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @dyn_access(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
func @dyn_access(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  //  CHECK: %{{.*}} = stencil.load %{{.*}}([-1, -2, 0] : [65, 66, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<66x68x60xf64>
  %2 = stencil.load %0 : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
  //  CHECK: %{{.*}} = stencil.apply (%{{.*}} = %{{.*}} : !stencil.temp<66x68x60xf64>) -> !stencil.temp<64x64x60xf64> {
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %cst = arith.constant 0 : index
    %4 = stencil.dyn_access %arg2(%cst, %cst, %cst) in [-1, -2, 0] : [1, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = stencil.store_result %4 : (f64) -> !stencil.result<f64>
    stencil.return %5 : !stencil.result<f64>
  //  CHECK: } to ([0, 0, 0] : [64, 64, 60])
  }
  //  CHECK: stencil.store %{{.*}} to %{{.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %3 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @combine
func @combine(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  // CHECK: %{{.*}} = stencil.load %{{.*}}([0, 0, 0] : [64, 64, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<64x64x60xf64>
  %2 = stencil.load %0 : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %6 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  // CHECK: } to ([0, 0, 0] : [32, 64, 60])
  } 
  %4 = stencil.apply (%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %6 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  // CHECK: } to ([32, 0, 0] : [64, 64, 60])
  } 
  // CHECK: %{{.*}} = stencil.combine 0 at 32 lower = (%{{.*}} : !stencil.temp<32x64x60xf64>) upper = (%{{.*}} : !stencil.temp<32x64x60xf64>) ([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64>
  %5 = stencil.combine 0 at 32 lower = (%3 : !stencil.temp<?x?x?xf64>) upper = (%4 : !stencil.temp<?x?x?xf64>) : !stencil.temp<?x?x?xf64>
  stencil.store %5 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @unroll(%{{.*}}: !stencil.field<?x?x?xf64>, %{{.*}}: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
func @unroll(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  //  CHECK: %{{.*}} = stencil.load %{{.*}}([-1, -2, 0] : [65, 66, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<66x68x60xf64>
  %2 = stencil.load %0 : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
  //  CHECK: %{{.*}} = stencil.apply (%{{.*}} = %{{.*}} : !stencil.temp<66x68x60xf64>) -> !stencil.temp<64x64x60xf64> {
  %3 = stencil.apply (%arg2 = %2 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %4 = stencil.access %arg2 [-1, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %5 = stencil.access %arg2 [1, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %6 = arith.addf %4, %5 : f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    %8 = stencil.access %arg2 [-1, 3, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %9 = stencil.access %arg2 [1, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %10 = arith.addf %8, %9 : f64
    %11 = stencil.store_result %10 : (f64) -> !stencil.result<f64>
    stencil.return unroll [1, 2, 1] %7, %11 : !stencil.result<f64>, !stencil.result<f64>
  //  CHECK: } to ([0, 0, 0] : [64, 64, 60])
  }
  //  CHECK: stencil.store %{{.*}} to %{{.*}}([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x60xf64>
  stencil.store %3 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @buffer
// CHECKEXT-LABEL: func @buffer
func @buffer(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %2 = stencil.apply -> !stencil.temp<?x?x?xf64> {
    %cst = arith.constant 1.0 : f64
    %10 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
    stencil.return %10 : !stencil.result<f64>
  // CHECK: } to ([0, 0, 0] : [64, 64, 60])
  // CHECKEXT: } to ([0, 0, 0] : [64, 64, 60])
  } 
  %3 = stencil.apply -> !stencil.temp<?x?x?xf64> {
    %cst = arith.constant 1.0 : f64
    %10 = stencil.store_result %cst : (f64) -> !stencil.result<f64>
    stencil.return %10 : !stencil.result<f64>
  // CHECK: } to ([32, 0, 0] : [64, 64, 60])
  // CHECKEXT: } to ([32, 0, 0] : [64, 64, 60])
  } 
  // CHECK: %{{.*}} = stencil.combine 0 at 32 lower = (%{{.*}} : !stencil.temp<64x64x60xf64>) upper = (%{{.*}} : !stencil.temp<64x64x60xf64>) upperext = (%{{.*}} : !stencil.temp<32x64x60xf64>) ([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64>, !stencil.temp<64x64x60xf64>
  // CHECKEXT: %{{.*}} = stencil.combine 0 at 32 lower = (%{{.*}} : !stencil.temp<64x64x60xf64>) upper = (%{{.*}} : !stencil.temp<64x64x60xf64>) upperext = (%{{.*}} : !stencil.temp<32x64x60xf64>) ([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64>, !stencil.temp<64x64x60xf64>
  %5:2 = stencil.combine 0 at 32 
    lower = (%2 : !stencil.temp<?x?x?xf64>) 
    upper = (%2 : !stencil.temp<?x?x?xf64>)
    upperext = (%3 : !stencil.temp<?x?x?xf64>) : !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>
  // CHECK: %{{.*}} = stencil.buffer %{{.*}}([0, 0, 0] : [64, 64, 60]) : (!stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64>
  // CHECKEXT: %{{.*}} = stencil.buffer %{{.*}}([0, 0, 0] : [64, 64, 60]) : (!stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64>
  %6 = stencil.buffer %5#0 : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  // CHECK: %{{.*}} = stencil.buffer %{{.*}}([48, 0, 0] : [64, 64, 60]) : (!stencil.temp<64x64x60xf64>) -> !stencil.temp<16x64x60xf64>
  // CHECKEXT:  %{{.*}} = stencil.buffer %{{.*}}([32, 0, 0] : [64, 64, 60]) : (!stencil.temp<64x64x60xf64>) -> !stencil.temp<32x64x60xf64>
  %7 = stencil.buffer %5#1 : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
  %8 = stencil.apply (%arg2 = %6 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %10 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %11 = stencil.store_result %10 : (f64) -> !stencil.result<f64>
    stencil.return %11 : !stencil.result<f64>
  }
  %9 = stencil.apply (%arg2 = %7 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %10 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %11 = stencil.store_result %10 : (f64) -> !stencil.result<f64>
    stencil.return %11 : !stencil.result<f64>
  }
  stencil.store %8 to %0([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  stencil.store %9 to %1([48, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}

// -----

// CHECK-LABEL: func @empty
func @empty(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %1 = stencil.cast %arg1([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  %2 = stencil.cast %arg2([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
  // CHECK: [[VAL0:%.*]] = stencil.load %{{.*}}([0, 0, 0] : [64, 64, 60]) : (!stencil.field<70x70x60xf64>) -> !stencil.temp<64x64x60xf64>
  %3 = stencil.load %0 : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
  // CHECK: [[VAL1:%.*]] = stencil.load %{{.*}} : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
  %4 = stencil.load %1 : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
  // CHECK: [[VAL2:%.*]] = stencil.apply (%{{.*}} = [[VAL0]] : !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64> {
  %5 = stencil.apply (%arg3 = %3 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %8 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %9 = stencil.store_result %8 : (f64) -> !stencil.result<f64>
    stencil.return %9 : !stencil.result<f64>
  // CHECK: } to ([0, 0, 0] : [64, 64, 60])
  }
  // CHECK: [[VAL3:%.*]] = stencil.apply (%{{.*}} = [[VAL1]] : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
  %6 = stencil.apply (%arg3 = %4 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
    %8 = stencil.access %arg3 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
    %9 = stencil.store_result %8 : (f64) -> !stencil.result<f64>
    stencil.return %9 : !stencil.result<f64>
  }
  // CHECK: %{{.*}} = stencil.combine 0 at 77 lower = ([[VAL2]] : !stencil.temp<64x64x60xf64>) upper = ([[VAL3]] : !stencil.temp<?x?x?xf64>) ([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64>
  %7 = stencil.combine 0 at 77 lower = (%5 : !stencil.temp<?x?x?xf64>) upper = (%6 : !stencil.temp<?x?x?xf64>) : !stencil.temp<?x?x?xf64>
  stencil.store %7 to %2([0, 0, 0] : [64, 64, 60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
  return
}

