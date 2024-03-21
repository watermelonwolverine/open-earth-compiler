// RUN: oec-opt %s -split-input-file --convert-stencil-to-std | FileCheck %s

// CHECK-LABEL: @func_lowering
// CHECK: (%{{.*}}: memref<?x?x?xf64>) {
func @func_lowering(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  // CHECK: %{{.*}} = memref_cast %{{.*}} : memref<?x?x?xf64> to memref<777x77x7xf64>
  %0 = stencil.cast %arg0 ([0, 0, 0]:[7, 77, 777]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<7x77x777xf64>
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]*]] = affine_map<(d0) -> (d0)>
// CHECK: [[MAP1:#map[0-9]*]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: @parallel_loop
func @parallel_loop(%arg0 : f64) attributes {stencil.program} {
  // CHECK-DAG: [[C0_1:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C0_2:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C0_3:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C1_1:%.*]] = arith.constant 1 : index
  // CHECK-DAG: [[C1_2:%.*]] = arith.constant 1 : index
  // CHECK-DAG: [[C1_3:%.*]] = arith.constant 1 : index
  // CHECK-DAG: [[C7:%.*]] = arith.constant 7 : index
  // CHECK-DAG: [[C77:%.*]] = arith.constant 77 : index
  // CHECK-DAG: [[C777:%.*]] = arith.constant 777 : index
  // CHECK: scf.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) = ([[C0_1]], [[C0_2]], [[C0_3]]) to ([[C7]], [[C77]], [[C777]]) step ([[C1_1]], [[C1_2]], [[C1_3]]) {  
  %0 = stencil.apply (%arg1 = %arg0 : f64) -> !stencil.temp<7x77x777xf64> {
    // CHECK-DAG:  [[IV0:%.*]] = affine.apply [[MAP0]]([[ARG0]])
    // CHECK-DAG:  [[IV1:%.*]] = affine.apply [[MAP0]]([[ARG1]])
    // CHECK-DAG:  [[IV2:%.*]] = affine.apply [[MAP0]]([[ARG2]])
    // CHECK-DAG:  [[IDX2:%.*]] = affine.apply [[MAP1]]([[IV0]], %{{.*}})
    // CHECK-DAG:  [[IDX1:%.*]] = affine.apply [[MAP1]]([[IV1]], %{{.*}})
    // CHECK-DAG:  [[IDX0:%.*]] = affine.apply [[MAP1]]([[IV2]], %{{.*}})
    // CHECK: store %{{.*}}, %{{.*}}{{\[}}[[IDX0]], [[IDX1]], [[IDX2]]] 
    %1 = stencil.store_result %arg1 : (f64) -> !stencil.result<f64>
    stencil.return %1 : !stencil.result<f64>
  } to ([0, 0, 0]:[7, 77, 777])
  %1 = stencil.buffer %0([0, 0, 0]:[7, 77, 777]) : (!stencil.temp<7x77x777xf64>) -> !stencil.temp<7x77x777xf64>
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]*]] = affine_map<(d0) -> (d0)>
// CHECK: [[MAP1:#map[0-9]*]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: @parallel_loop_unroll
func @parallel_loop_unroll(%arg0 : f64) attributes {stencil.program} {
  // CHECK-DAG: [[C0_1:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[C0_2:%.*]] = arith.constant 0 : index
  // CHECK-DAG: [[CM1:%.*]] = arith.constant -1 : index
  // CHECK-DAG: [[C1_1:%.*]] = arith.constant 1 : index
  // CHECK-DAG: [[C1_2:%.*]] = arith.constant 1 : index
  // CHECK-DAG: [[C2:%.*]] = arith.constant 2 : index
  // CHECK-DAG: [[C7:%.*]] = arith.constant 7 : index
  // CHECK-DAG: [[C77:%.*]] = arith.constant 77 : index
  // CHECK-DAG: [[C777:%.*]] = arith.constant 777 : index
  // CHECK-NEXT: scf.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) = ([[C0_1]], [[CM1]], [[C0_2]]) to ([[C7]], [[C77]], [[C777]]) step ([[C1_1]], [[C2]], [[C1_2]]) {  
  %0 = stencil.apply (%arg1 = %arg0 : f64) -> !stencil.temp<7x78x777xf64> {
    // CHECK-DAG:  [[IV1:%.*]] = affine.apply [[MAP0]]([[ARG1]])
    // CHECK-DAG:  [[U0O1:%.*]] = arith.constant 1 : index
    // CHECK-DAG:  [[U0IDX1:%.*]] = affine.apply [[MAP1]]([[IV1]], [[U0O1]])
    // CHECK: store %{{.*}}, %{{.*}}{{\[}}%{{.*}}, [[U0IDX1]], %{{.*}}]
    // CHECK-DAG:  [[U1O1:%.*]] = arith.constant 2 : index
    // CHECK-DAG:  [[U1IDX1:%.*]] = affine.apply [[MAP1]]([[IV1]], [[U1O1]])
    // CHECK: store %{{.*}}, %{{.*}}{{\[}}%{{.*}}, [[U1IDX1]], %{{.*}}]
    %1 = stencil.store_result %arg1 : (f64) -> !stencil.result<f64>
    %2 = stencil.store_result %arg1 : (f64) -> !stencil.result<f64>
    stencil.return unroll [1, 2, 1] %1, %2 : !stencil.result<f64>, !stencil.result<f64>
  } to ([0, -1, 0]:[7, 77, 777])
  %1 = stencil.buffer %0([0, -1, 0]:[7, 77, 777]) : (!stencil.temp<7x78x777xf64>) -> !stencil.temp<7x78x777xf64>
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]*]] = affine_map<(d0) -> (d0)>
// CHECK: [[MAP1:#map[0-9]*]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: @access_lowering
func @access_lowering(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0 ([0, 0, 0]:[10, 10, 10]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<10x10x10xf64>
  // CHECK: [[VIEW:%.*]] = subview %{{.*}}[0, 0, 0] [10, 10, 10] [1, 1, 1]
  %1 = stencil.load %0 ([0, 0, 0]:[10, 10, 10]) : (!stencil.field<10x10x10xf64>) -> !stencil.temp<10x10x10xf64>
  // CHECK: scf.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) =
  %2 = stencil.apply (%arg1 = %1 : !stencil.temp<10x10x10xf64>) -> !stencil.temp<7x7x7xf64> {
    // CHECK-DAG: [[IV0:%.*]] = affine.apply [[MAP0]]([[ARG0]])
    // CHECK-DAG: [[IV1:%.*]] = affine.apply [[MAP0]]([[ARG1]])
    // CHECK-DAG: [[IV2:%.*]] = affine.apply [[MAP0]]([[ARG2]])
    // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
    // CHECK-DAG: [[O0:%.*]] = affine.apply [[MAP1]]([[IV0]], [[C0]])
    // CHECK-DAG: [[C1:%.*]] = arith.constant 1 : index
    // CHECK-DAG: [[O1:%.*]] = affine.apply [[MAP1]]([[IV1]], [[C1]])
    // CHECK-DAG: [[C2:%.*]] = arith.constant 2 : index
    // CHECK-DAG: [[O2:%.*]] = affine.apply [[MAP1]]([[IV2]], [[C2]])
    // CHECK: %{{.*}} = load [[VIEW:%.*]]{{\[}}[[O2]], [[O1]], [[O0]]{{[]]}}
    %4 = stencil.access %arg1[0, 1, 2] : (!stencil.temp<10x10x10xf64>) -> f64
    %5 = stencil.store_result %4 : (f64) -> !stencil.result<f64>
    stencil.return %5 : !stencil.result<f64>
  } to ([0, 0, 0]:[7, 7, 7])
  %3 = stencil.buffer %2([0, 0, 0]:[7, 7, 7]) : (!stencil.temp<7x7x7xf64>) -> !stencil.temp<7x7x7xf64>
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]*]] = affine_map<(d0) -> (d0)>
// CHECK: [[MAP1:#map[0-9]*]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: @index_lowering
func @index_lowering(%arg0 : f64) attributes {stencil.program} {
  // CHECK: scf.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) =
  %0 = stencil.apply (%arg1 = %arg0 : f64) -> !stencil.temp<7x7x7xf64> {
    // CHECK-DAG: [[IV2:%.*]] = affine.apply [[MAP0]]([[ARG2]])
    // CHECK-DAG: [[C0:%.*]] = arith.constant 2 : index
    // CHECK-DAG: [[O0:%.*]] = affine.apply [[MAP1]]([[IV2]], [[C0]])
    %2 = stencil.index 2 [0, 1, 2] : index
    %cst = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f64
    %3 = cmpi "slt", %2, %cst : index
    %4 = select %3, %arg1, %cst_0 : f64
    %5 = stencil.store_result %4 : (f64) -> !stencil.result<f64>
    stencil.return %5 : !stencil.result<f64>
  } to ([0, 0, 0]:[7, 7, 7])
  %1 = stencil.buffer %0([0, 0, 0]:[7, 7, 7]) : (!stencil.temp<7x7x7xf64>) -> !stencil.temp<7x7x7xf64>
  return
}

// -----

// CHECK-LABEL: @return_lowering
func @return_lowering(%arg0: f64) attributes {stencil.program} {
  // CHECK: scf.parallel (%{{.*}}, %{{.*}}, %{{.*}}) =
  %0:2 = stencil.apply (%arg1 = %arg0 : f64) -> (!stencil.temp<7x7x7xf64>, !stencil.temp<7x7x7xf64>) {
    // CHECK-COUNT-2: store %{{.*}}, %{{.*}}{{\[}}%{{.*}}, %{{.*}}, %{{.*}} : memref<7x7x7xf64> 
    %3 = stencil.store_result %arg1 : (f64) -> !stencil.result<f64>
    %4 = stencil.store_result %arg1 : (f64) -> !stencil.result<f64>
    stencil.return %3, %4 : !stencil.result<f64>, !stencil.result<f64>
  } to ([0, 0, 0]:[7, 7, 7])
  %1 = stencil.buffer %0#0([0, 0, 0]:[7, 7, 7]) : (!stencil.temp<7x7x7xf64>) -> !stencil.temp<7x7x7xf64>
  %2 = stencil.buffer %0#1([0, 0, 0]:[7, 7, 7]) : (!stencil.temp<7x7x7xf64>) -> !stencil.temp<7x7x7xf64>
  return
}

// -----

// CHECK-LABEL: @if_lowering
func @if_lowering(%arg0: f64) attributes {stencil.program} {
  // CHECK: scf.parallel (%{{.*}}, %{{.*}}, %{{.*}}) =
  %0:2 = stencil.apply (%arg1 = %arg0 : f64) -> (!stencil.temp<7x7x7xf64>, !stencil.temp<7x7x7xf64>) {
    %cond = arith.constant 1 : i1
    // CHECK: %{{.*}} = scf.if %{{.*}} -> (f64) { 
    %1, %2 = scf.if %cond -> (!stencil.result<f64>, f64) {
      // CHECK: store %{{.*}}, %{{.*}}{{\[}}%{{.*}}, %{{.*}}, %{{.*}}] : memref<7x7x7xf64> 
      // CHECK: scf.yield %{{.*}} : f64 
      %3 = stencil.store_result %arg1 : (f64) -> !stencil.result<f64>
      scf.yield %3, %arg1 : !stencil.result<f64>, f64
    } else {
      // CHECK: } else { 
      // CHECK-NEXT: scf.yield %{{.*}} : f64
      %3 = stencil.store_result : () -> !stencil.result<f64>
      scf.yield %3, %arg1 : !stencil.result<f64>, f64
    }
    %4 = stencil.store_result %2 : (f64) -> !stencil.result<f64>
    stencil.return %1, %4 : !stencil.result<f64>, !stencil.result<f64>
  } to ([0, 0, 0]:[7, 7, 7])
  %1 = stencil.buffer %0#0([0, 0, 0]:[7, 7, 7]) : (!stencil.temp<7x7x7xf64>) -> !stencil.temp<7x7x7xf64>
  %2 = stencil.buffer %0#1([0, 0, 0]:[7, 7, 7]) : (!stencil.temp<7x7x7xf64>) -> !stencil.temp<7x7x7xf64>
  return
}

// -----

// CHECK-LABEL: @load_lowering
func @load_lowering(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0 ([0, 0, 0]:[11, 12, 13]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<11x12x13xf64>
  // CHECK: %{{.*}} = subview %{{.*}}[3, 2, 1] [9, 9, 9] [1, 1, 1] : memref<13x12x11xf64> to memref<9x9x9xf64, #map{{[0-9]*}}>
  %1 = stencil.load %0 ([1, 2, 3]:[10, 11, 12]) : (!stencil.field<11x12x13xf64>) -> !stencil.temp<9x9x9xf64>
  return
}

// -----

// CHECK-LABEL: @store_lowering
func @store_lowering(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0 ([0, 0, 0]:[10, 10, 10]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<10x10x10xf64>
  // CHECK: [[VIEW:%.*]] = subview %{{.*}}[3, 2, 1] [7, 7, 7] [1, 1, 1] : memref<10x10x10xf64> to memref<7x7x7xf64, #map{{[0-9]*}}>
  %cst = arith.constant 1.0 : f64
  %1 = stencil.apply (%arg1 = %cst : f64) -> !stencil.temp<7x7x7xf64> {
    // CHECK: store %{{.*}} [[VIEW]]
    %2 = stencil.store_result %arg1 : (f64) -> !stencil.result<f64>
    stencil.return %2 : !stencil.result<f64>
  } to ([0, 0, 0]:[7, 7, 7]) 
  stencil.store %1 to %0 ([1, 2, 3]:[8, 9, 10]) : !stencil.temp<7x7x7xf64> to !stencil.field<10x10x10xf64>
  return
}

// -----

// CHECK-LABEL: @if_lowering
func @if_lowering(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0 ([0, 0, 0]:[10, 10, 10]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<10x10x10xf64>
  %1 = stencil.load %0 ([0, 0, 0]:[10, 10, 10]) : (!stencil.field<10x10x10xf64>) -> !stencil.temp<10x10x10xf64>
  %2 = stencil.apply (%arg1 = %1 : !stencil.temp<10x10x10xf64>) -> !stencil.temp<7x7x7xf64> {
    %cst = arith.constant 1 : i1
    // CHECK: [[RES:%.*]] = scf.if %{{.*}} -> (f64) {
    %3 = scf.if %cst -> (f64) {
      // CHECK: [[IF:%.*]] = load
      %5 = stencil.access %arg1[0, 1, 2] : (!stencil.temp<10x10x10xf64>) -> f64
      // CHECK: scf.yield [[IF]] : f64
      scf.yield %5 : f64
    // CHECK: } else {
    } else {
      // CHECK: [[ELSE:%.*]] = load
      %5 = stencil.access %arg1[0, 2, 1] : (!stencil.temp<10x10x10xf64>) -> f64
      // CHECK: scf.yield [[ELSE]] : f64
      scf.yield %5 : f64
    }
    // CHECK: store [[RES]]
    %5 = stencil.store_result %3 : (f64) -> !stencil.result<f64>
    stencil.return %5 : !stencil.result<f64>
  } to ([0, 0, 0]:[7, 7, 7])
  %3 = stencil.buffer %2([0, 0, 0]:[7, 7, 7]) : (!stencil.temp<7x7x7xf64>) -> !stencil.temp<7x7x7xf64>
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]*]] = affine_map<(d0) -> (d0)>
// CHECK: [[MAP1:#map[0-9]*]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: @lowerdim
// CHECK: (%{{.*}}: memref<?x?xf64>) {
func @lowerdim(%arg0: !stencil.field<?x?x0xf64>) attributes {stencil.program} {
  // CHECK: %{{.*}} = memref_cast %{{.*}} : memref<?x?xf64> to memref<11x10xf64>
  %0 = stencil.cast %arg0 ([0, 0, 0]:[10, 11, 12]) : (!stencil.field<?x?x0xf64>) -> !stencil.field<10x11x0xf64>
  // CHECK: [[VIEW:%.*]] = subview %{{.*}}[0, 0] [8, 7] [1, 1] : memref<11x10xf64> to memref<8x7xf64, #map{{[0-9]*}}>
  %1 = stencil.load %0 ([0, 0, 0]:[7, 8, 9]) : (!stencil.field<10x11x0xf64>) -> !stencil.temp<7x8x0xf64>
  // CHECK: scf.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) =
  %2 = stencil.apply (%arg1 = %1 : !stencil.temp<7x8x0xf64>) -> !stencil.temp<7x7x7xf64> {
    // CHECK-DAG: [[IV0:%.*]] = affine.apply [[MAP0]]([[ARG0]])
    // CHECK-DAG: [[IV1:%.*]] = affine.apply [[MAP0]]([[ARG1]])
    // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
    // CHECK-DAG: [[O0:%.*]] = affine.apply [[MAP1]]([[IV0]], [[C0]])
    // CHECK-DAG: [[C1:%.*]] = arith.constant 1 : index
    // CHECK-DAG: [[O1:%.*]] = affine.apply [[MAP1]]([[IV1]], [[C1]])
    // CHECK: %{{.*}} = load [[VIEW:%.*]]{{\[}}[[O1]], [[O0]]{{[]]}}
    %3 = stencil.access %arg1[0, 1, 2]: (!stencil.temp<7x8x0xf64>) -> f64
    %4 = stencil.store_result %3 : (f64) -> !stencil.result<f64>
    stencil.return unroll %4 : !stencil.result<f64>
  } to ([0, 0, 0]:[7, 7, 7])
  %3 = stencil.buffer %2([0, 0, 0]:[7, 7, 7]) : (!stencil.temp<7x7x7xf64>) -> !stencil.temp<7x7x7xf64>
  return
}

// -----

// CHECK: [[MAP0:#map[0-9]*]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: @dyn_access_lowering
func @dyn_access_lowering(%arg0: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
  %0 = stencil.cast %arg0([0, 0, 0] : [10, 10, 10]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<10x10x10xf64>
  // CHECK: %{{.*}} = subview %{{.*}}[0, 0, 0] [10, 10, 10] [1, 1, 1]
  %1 = stencil.load %0([0, 0, 0] : [10, 10, 10]) : (!stencil.field<10x10x10xf64>) -> !stencil.temp<10x10x10xf64>
  // CHECK: scf.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) =
  %2 = stencil.apply (%arg1 = %1 : !stencil.temp<10x10x10xf64>) -> !stencil.temp<7x7x7xf64> {
    // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
    // CHECK-DAG: [[C1:%.*]] = arith.constant 1 : index
    // CHECK-DAG: [[C2:%.*]] = arith.constant 2 : index
    // CHECK-DAG: [[IDX0:%.*]] = affine.apply [[MAP0]]([[C0]], %{{.*}})
    // CHECK-DAG: [[IDX1:%.*]] = affine.apply [[MAP0]]([[C1]], %{{.*}})
    // CHECK-DAG: [[IDX2:%.*]] = affine.apply [[MAP0]]([[C2]], %{{.*}})
    // CHECK: %{{.*}} = load %{{.*}}{{\[}}[[IDX2]], [[IDX1]], [[IDX0]]{{[]]}}
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %3 = stencil.dyn_access %arg1(%c0, %c1, %c2) in [0, 0, 0] : [0, 1, 2] : (!stencil.temp<10x10x10xf64>) -> f64
    %4 = stencil.store_result %3 : (f64) -> !stencil.result<f64>
    stencil.return unroll %4 : !stencil.result<f64>
  } to ([0, 0, 0] : [7, 7, 7])
  %3 = stencil.buffer %2([0, 0, 0]:[7, 7, 7]) : (!stencil.temp<7x7x7xf64>) -> !stencil.temp<7x7x7xf64>
  return
}

// -----

// CHECK: [[MAP1:#map[0-9]*]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: @alloc_temp
func @alloc_temp(%arg0 : f64) attributes {stencil.program} {
  // CHECK: [[TEMP1:%.*]] = gpu.alloc () : memref<7x7x7xf64>
  // CHECK: [[TEMP2:%.*]] = gpu.alloc () : memref<6x6x6xf64>
  // CHECK: scf.parallel ([[ARG0:%.*]], [[ARG1:%.*]], [[ARG2:%.*]]) =
  %0,%1 = stencil.apply (%arg1 = %arg0 : f64) -> (!stencil.temp<7x7x7xf64>, !stencil.temp<7x7x7xf64>) {
    // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : index
    // CHECK-DAG: [[C1:%.*]] = arith.constant 0 : index
    // CHECK-DAG: [[C2:%.*]] = arith.constant 0 : index
    // CHECK-DAG: [[C3:%.*]] = arith.constant -1 : index
    // CHECK-DAG: [[C4:%.*]] = arith.constant -1 : index
    // CHECK-DAG: [[C5:%.*]] = arith.constant -1 : index
    // CHECK-DAG: [[IDX0:%.*]] = affine.apply [[MAP1]](%{{.*}}, [[C0]])
    // CHECK-DAG: [[IDX1:%.*]] = affine.apply [[MAP1]](%{{.*}}, [[C1]])
    // CHECK-DAG: [[IDX2:%.*]] = affine.apply [[MAP1]](%{{.*}}, [[C2]])
    // CHECK-DAG: [[IDX3:%.*]] = affine.apply [[MAP1]](%{{.*}}, [[C3]])
    // CHECK-DAG: [[IDX4:%.*]] = affine.apply [[MAP1]](%{{.*}}, [[C4]])
    // CHECK-DAG: [[IDX5:%.*]] = affine.apply [[MAP1]](%{{.*}}, [[C5]])
    // CHECK-DAG: store %{{.*}}, [[TEMP1]]{{\[}}[[IDX2]], [[IDX1]], [[IDX0]]] : memref<7x7x7xf64> 
    // CHECK-DAG: store %{{.*}}, [[TEMP2]]{{\[}}[[IDX5]], [[IDX4]], [[IDX3]]] : memref<6x6x6xf64> 
    %6 = stencil.store_result %arg1 : (f64) -> !stencil.result<f64>
    %7 = stencil.store_result %arg1 : (f64) -> !stencil.result<f64>
    stencil.return %6, %7 : !stencil.result<f64>, !stencil.result<f64>
  } to ([0, 0, 0]:[7, 7, 7]) 
  %2 = stencil.buffer %0([0, 0, 0]:[7, 7, 7]) : (!stencil.temp<7x7x7xf64>) -> !stencil.temp<7x7x7xf64>
  %3 = stencil.buffer %1([1, 1, 1]:[7, 7, 7]) : (!stencil.temp<7x7x7xf64>) -> !stencil.temp<6x6x6xf64>
  // CHECK: gpu.dealloc [[TEMP2]] : memref<6x6x6xf64>
  // CHECK: [[TEMP3:%.*]] = gpu.alloc () : memref<7x7x7xf64>
  %4 = stencil.apply (%arg1 = %2 : !stencil.temp<7x7x7xf64>) -> !stencil.temp<7x7x7xf64> {
    // CHECK: load [[TEMP1]]
    // CHECK: store %{{.*}}, [[TEMP3]]    
    %6 = stencil.access %arg1[0,0,0] : (!stencil.temp<7x7x7xf64>) -> f64
    %7 = stencil.store_result %6 : (f64) -> !stencil.result<f64>
    stencil.return %7 : !stencil.result<f64>
  } to ([0, 0, 0]:[7, 7, 7])
  %5 = stencil.buffer %4([0, 0, 0]:[7, 7, 7]) : (!stencil.temp<7x7x7xf64>) -> !stencil.temp<7x7x7xf64>
  // CHECK: gpu.dealloc [[TEMP1]] : memref<7x7x7xf64>
  // CHECK: gpu.dealloc [[TEMP3]] : memref<7x7x7xf64>
  return
}
