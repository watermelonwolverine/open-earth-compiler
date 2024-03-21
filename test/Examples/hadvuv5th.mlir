

module {
  func @hadvuv5th(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>, %arg2: !stencil.field<?x?x?xf64>, %arg3: !stencil.field<?x?x?xf64>, %arg4: !stencil.field<0x?x0xf64>, %arg5: !stencil.field<0x?x0xf64>, %arg6: !stencil.field<0x?x0xf64>, %arg7: !stencil.field<0x?x0xf64>) attributes {stencil.program} {
    %0 = stencil.cast %arg0([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %1 = stencil.cast %arg1([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %2 = stencil.cast %arg2([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %3 = stencil.cast %arg3([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<72x72x72xf64>
    %4 = stencil.cast %arg4([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<0x?x0xf64>) -> !stencil.field<0x72x0xf64>
    %5 = stencil.cast %arg5([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<0x?x0xf64>) -> !stencil.field<0x72x0xf64>
    %6 = stencil.cast %arg6([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<0x?x0xf64>) -> !stencil.field<0x72x0xf64>
    %7 = stencil.cast %arg7([-4, -4, -4] : [68, 68, 68]) : (!stencil.field<0x?x0xf64>) -> !stencil.field<0x72x0xf64>
    %8 = stencil.load %0 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %9 = stencil.load %1 : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
    %10 = stencil.load %4 : (!stencil.field<0x72x0xf64>) -> !stencil.temp<0x?x0xf64>
    %11 = stencil.load %5 : (!stencil.field<0x72x0xf64>) -> !stencil.temp<0x?x0xf64>
    %12 = stencil.load %6 : (!stencil.field<0x72x0xf64>) -> !stencil.temp<0x?x0xf64>
    %13 = stencil.load %7 : (!stencil.field<0x72x0xf64>) -> !stencil.temp<0x?x0xf64>
    %14:2 = stencil.apply (%arg8 = %8 : !stencil.temp<?x?x?xf64>, %arg9 = %10 : !stencil.temp<0x?x0xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 3.000000e+00 : f64
      %22 = divf %cst, %cst_0 : f64
      %23 = stencil.access %arg8 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = stencil.access %arg8 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = arith.addf %23, %24 : f64
      %27 = arith.addf %26, %25 : f64
      %28 = arith.mulf %27, %22 : f64
      %29 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %30 = arith.mulf %28, %29 : f64
      %31 = stencil.store_result %28 : (f64) -> !stencil.result<f64>
      %32 = stencil.store_result %30 : (f64) -> !stencil.result<f64>
      stencil.return %31, %32 : !stencil.result<f64>, !stencil.result<f64>
    }
    %15:2 = stencil.apply (%arg8 = %9 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 6.371229e+06 : f64
      %22 = divf %cst, %cst_0 : f64
      %cst_1 = arith.constant 2.500000e-01 : f64
      %23 = stencil.access %arg8 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = stencil.access %arg8 [1, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = stencil.access %arg8 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = arith.addf %23, %24 : f64
      %28 = arith.addf %25, %26 : f64
      %29 = arith.addf %27, %28 : f64
      %30 = arith.mulf %29, %cst_1 : f64
      %31 = arith.mulf %30, %22 : f64
      %32 = stencil.store_result %30 : (f64) -> !stencil.result<f64>
      %33 = stencil.store_result %31 : (f64) -> !stencil.result<f64>
      stencil.return %32, %33 : !stencil.result<f64>, !stencil.result<f64>
    }
    %16 = stencil.apply (%arg8 = %8 : !stencil.temp<?x?x?xf64>, %arg9 = %14#1 : !stencil.temp<?x?x?xf64>, %arg10 = %15#1 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = arith.constant 4.8828125E-4 : f64
      %cst_0 = arith.constant 7.32421875E-4 : f64
      %cst_1 = arith.constant 0.000000e+00 : f64
      %cst_2 = arith.constant 1.000000e+00 : f64
      %cst_3 = arith.constant -1.000000e+00 : f64
      %cst_4 = arith.constant 2.000000e+00 : f64
      %cst_5 = arith.constant 3.000000e+00 : f64
      %cst_6 = arith.constant 4.000000e+00 : f64
      %cst_7 = arith.constant 2.000000e+01 : f64
      %cst_8 = arith.constant 3.000000e+01 : f64
      %22 = divf %cst_2, %cst_8 : f64
      %23 = divf %cst_3, %cst_6 : f64
      %24 = divf %cst_3, %cst_5 : f64
      %25 = divf %cst_3, %cst_4 : f64
      %26 = divf %cst_2, %cst_7 : f64
      %27 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = stencil.access %arg8 [-2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = stencil.access %arg8 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = stencil.access %arg8 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %32 = stencil.access %arg8 [2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = cmpf "ogt", %27, %cst_1 : f64
      %34 = arith.mulf %24, %28 : f64
      %35 = stencil.access %arg8 [-3, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %36 = arith.mulf %23, %29 : f64
      %37 = arith.addf %34, %30 : f64
      %38 = arith.mulf %25, %31 : f64
      %39 = arith.mulf %26, %32 : f64
      %40 = arith.mulf %22, %35 : f64
      %41 = arith.addf %36, %38 : f64
      %42 = arith.addf %39, %40 : f64
      %43 = arith.addf %37, %41 : f64
      %44 = arith.addf %42, %43 : f64
      %45 = arith.mulf %44, %27 : f64
      %46 = stencil.access %arg8 [3, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %47 = arith.mulf %26, %29 : f64
      %48 = arith.mulf %25, %30 : f64
      %49 = arith.addf %34, %31 : f64
      %50 = arith.mulf %23, %32 : f64
      %51 = arith.mulf %22, %46 : f64
      %52 = arith.addf %47, %48 : f64
      %53 = arith.addf %50, %51 : f64
      %54 = arith.addf %49, %52 : f64
      %55 = arith.addf %53, %54 : f64
      %56 = negf %27 : f64
      %57 = arith.mulf %56, %55 : f64
      %58 = select %33, %45, %57 : f64
      %59 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %60 = stencil.access %arg8 [0, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %61 = stencil.access %arg8 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %62 = stencil.access %arg8 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %63 = stencil.access %arg8 [0, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %64 = cmpf "ogt", %59, %cst_1 : f64
      %65 = stencil.access %arg8 [0, -3, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %66 = arith.mulf %23, %60 : f64
      %67 = arith.addf %34, %61 : f64
      %68 = arith.mulf %25, %62 : f64
      %69 = arith.mulf %26, %63 : f64
      %70 = arith.mulf %22, %65 : f64
      %71 = arith.addf %66, %68 : f64
      %72 = arith.addf %69, %70 : f64
      %73 = arith.addf %67, %71 : f64
      %74 = arith.addf %72, %73 : f64
      %75 = arith.mulf %74, %59 : f64
      %76 = stencil.access %arg8 [0, 3, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %77 = arith.mulf %26, %60 : f64
      %78 = arith.mulf %25, %61 : f64
      %79 = arith.addf %34, %62 : f64
      %80 = arith.mulf %23, %63 : f64
      %81 = arith.mulf %22, %76 : f64
      %82 = arith.addf %77, %78 : f64
      %83 = arith.addf %80, %81 : f64
      %84 = arith.addf %79, %82 : f64
      %85 = arith.addf %83, %84 : f64
      %86 = negf %59 : f64
      %87 = arith.mulf %86, %85 : f64
      %88 = select %64, %75, %87 : f64
      %89 = arith.mulf %58, %cst : f64
      %90 = arith.mulf %88, %cst_0 : f64
      %91 = arith.addf %89, %90 : f64
      %92 = stencil.store_result %91 : (f64) -> !stencil.result<f64>
      stencil.return %92 : !stencil.result<f64>
    }
    %17 = stencil.apply (%arg8 = %16 : !stencil.temp<?x?x?xf64>, %arg9 = %8 : !stencil.temp<?x?x?xf64>, %arg10 = %15#0 : !stencil.temp<?x?x?xf64>, %arg11 = %12 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %22 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg11 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %25 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = arith.mulf %22, %23 : f64
      %27 = arith.mulf %26, %24 : f64
      %28 = arith.addf %27, %25 : f64
      %29 = stencil.store_result %28 : (f64) -> !stencil.result<f64>
      stencil.return %29 : !stencil.result<f64>
    }
    %18:2 = stencil.apply (%arg8 = %8 : !stencil.temp<?x?x?xf64>, %arg9 = %11 : !stencil.temp<0x?x0xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = arith.constant 2.500000e-01 : f64
      %22 = stencil.access %arg8 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg8 [-1, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %24 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = stencil.access %arg8 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = arith.addf %22, %23 : f64
      %27 = arith.addf %24, %25 : f64
      %28 = arith.addf %26, %27 : f64
      %29 = arith.mulf %28, %cst : f64
      %30 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %31 = arith.mulf %29, %30 : f64
      %32 = stencil.store_result %29 : (f64) -> !stencil.result<f64>
      %33 = stencil.store_result %31 : (f64) -> !stencil.result<f64>
      stencil.return %32, %33 : !stencil.result<f64>, !stencil.result<f64>
    }
    %19:2 = stencil.apply (%arg8 = %9 : !stencil.temp<?x?x?xf64>) -> (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 6.371229e+06 : f64
      %22 = divf %cst, %cst_0 : f64
      %cst_1 = arith.constant 3.000000e+00 : f64
      %23 = divf %cst, %cst_1 : f64
      %24 = stencil.access %arg8 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %26 = stencil.access %arg8 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %27 = arith.addf %24, %25 : f64
      %28 = arith.addf %27, %26 : f64
      %29 = arith.mulf %28, %23 : f64
      %30 = arith.mulf %29, %22 : f64
      %31 = stencil.store_result %29 : (f64) -> !stencil.result<f64>
      %32 = stencil.store_result %30 : (f64) -> !stencil.result<f64>
      stencil.return %31, %32 : !stencil.result<f64>, !stencil.result<f64>
    }
    %20 = stencil.apply (%arg8 = %9 : !stencil.temp<?x?x?xf64>, %arg9 = %18#1 : !stencil.temp<?x?x?xf64>, %arg10 = %19#1 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
      %cst = arith.constant 4.8828125E-4 : f64
      %cst_0 = arith.constant 7.32421875E-4 : f64
      %cst_1 = arith.constant 0.000000e+00 : f64
      %cst_2 = arith.constant 1.000000e+00 : f64
      %cst_3 = arith.constant -1.000000e+00 : f64
      %cst_4 = arith.constant 2.000000e+00 : f64
      %cst_5 = arith.constant 3.000000e+00 : f64
      %cst_6 = arith.constant 4.000000e+00 : f64
      %cst_7 = arith.constant 2.000000e+01 : f64
      %cst_8 = arith.constant 3.000000e+01 : f64
      %22 = divf %cst_2, %cst_8 : f64
      %23 = divf %cst_3, %cst_6 : f64
      %24 = divf %cst_3, %cst_5 : f64
      %25 = divf %cst_3, %cst_4 : f64
      %26 = divf %cst_2, %cst_7 : f64
      %27 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %28 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %29 = stencil.access %arg8 [-2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %30 = stencil.access %arg8 [-1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %31 = stencil.access %arg8 [1, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %32 = stencil.access %arg8 [2, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %33 = cmpf "ogt", %27, %cst_1 : f64
      %34 = arith.mulf %24, %28 : f64
      %35 = stencil.access %arg8 [-3, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %36 = arith.mulf %23, %29 : f64
      %37 = arith.addf %34, %30 : f64
      %38 = arith.mulf %25, %31 : f64
      %39 = arith.mulf %26, %32 : f64
      %40 = arith.mulf %22, %35 : f64
      %41 = arith.addf %36, %38 : f64
      %42 = arith.addf %39, %40 : f64
      %43 = arith.addf %37, %41 : f64
      %44 = arith.addf %42, %43 : f64
      %45 = arith.mulf %44, %27 : f64
      %46 = stencil.access %arg8 [3, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %47 = arith.mulf %26, %29 : f64
      %48 = arith.mulf %25, %30 : f64
      %49 = arith.addf %34, %31 : f64
      %50 = arith.mulf %23, %32 : f64
      %51 = arith.mulf %22, %46 : f64
      %52 = arith.addf %47, %48 : f64
      %53 = arith.addf %50, %51 : f64
      %54 = arith.addf %49, %52 : f64
      %55 = arith.addf %53, %54 : f64
      %56 = negf %27 : f64
      %57 = arith.mulf %56, %55 : f64
      %58 = select %33, %45, %57 : f64
      %59 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %60 = stencil.access %arg8 [0, -2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %61 = stencil.access %arg8 [0, -1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %62 = stencil.access %arg8 [0, 1, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %63 = stencil.access %arg8 [0, 2, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %64 = cmpf "ogt", %59, %cst_1 : f64
      %65 = stencil.access %arg8 [0, -3, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %66 = arith.addf %34, %61 : f64
      %67 = arith.mulf %23, %60 : f64
      %68 = arith.mulf %25, %62 : f64
      %69 = arith.mulf %26, %63 : f64
      %70 = arith.mulf %22, %65 : f64
      %71 = arith.addf %67, %68 : f64
      %72 = arith.addf %69, %70 : f64
      %73 = arith.addf %66, %71 : f64
      %74 = arith.addf %72, %73 : f64
      %75 = arith.mulf %74, %59 : f64
      %76 = stencil.access %arg8 [0, 3, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %77 = arith.addf %34, %62 : f64
      %78 = arith.mulf %26, %60 : f64
      %79 = arith.mulf %25, %61 : f64
      %80 = arith.mulf %23, %63 : f64
      %81 = arith.mulf %22, %76 : f64
      %82 = arith.addf %78, %79 : f64
      %83 = arith.addf %80, %81 : f64
      %84 = arith.addf %77, %82 : f64
      %85 = arith.addf %83, %84 : f64
      %86 = negf %59 : f64
      %87 = arith.mulf %86, %85 : f64
      %88 = select %64, %75, %87 : f64
      %89 = arith.mulf %58, %cst : f64
      %90 = arith.mulf %88, %cst_0 : f64
      %91 = arith.addf %89, %90 : f64
      %92 = stencil.store_result %91 : (f64) -> !stencil.result<f64>
      stencil.return %92 : !stencil.result<f64>
    }
    %21 = stencil.apply (%arg8 = %20 : !stencil.temp<?x?x?xf64>, %arg9 = %18#0 : !stencil.temp<?x?x?xf64>, %arg10 = %13 : !stencil.temp<0x?x0xf64>) -> !stencil.temp<?x?x?xf64> {
      %22 = stencil.access %arg9 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %23 = stencil.access %arg10 [0, 0, 0] : (!stencil.temp<0x?x0xf64>) -> f64
      %24 = stencil.access %arg8 [0, 0, 0] : (!stencil.temp<?x?x?xf64>) -> f64
      %25 = arith.mulf %22, %22 : f64
      %26 = arith.mulf %25, %23 : f64
      %27 = subf %24, %26 : f64
      %28 = stencil.store_result %27 : (f64) -> !stencil.result<f64>
      stencil.return %28 : !stencil.result<f64>
    }
    stencil.store %17 to %2([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    stencil.store %21 to %3([0, 0, 0] : [64, 64, 64]) : !stencil.temp<?x?x?xf64> to !stencil.field<72x72x72xf64>
    return
  }
}
