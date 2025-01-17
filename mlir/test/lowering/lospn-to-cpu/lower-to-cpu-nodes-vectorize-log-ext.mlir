// RUN: %optcall --vectorize-lospn-nodes %s | FileCheck %s

module  {
  func @vec_task_0(%arg0: memref<?x4xf32>, %arg1: memref<1x?xf64>) {
    %c0 = constant 0 : index
    %0 = memref.dim %arg0, %c0 : memref<?x4xf32>
    %c8 = constant 8 : index
    %1 = remi_unsigned %0, %c8 : index
    %2 = subi %0, %1 : index
    %c0_0 = constant 0 : index
    %c8_1 = constant 8 : index
    scf.for %arg2 = %c0_0 to %2 step %c8_1 {
      %3 = "lo_spn.batch_read"(%arg0, %arg2) {staticIndex = 0 : ui32, vector_width = 4 : i32} : (memref<?x4xf32>, index) -> f32
      %4 = "lo_spn.batch_read"(%arg0, %arg2) {staticIndex = 1 : ui32, vector_width = 4 : i32} : (memref<?x4xf32>, index) -> f32
      %5 = "lo_spn.batch_read"(%arg0, %arg2) {staticIndex = 2 : ui32, vector_width = 4 : i32} : (memref<?x4xf32>, index) -> f32
      %6 = "lo_spn.batch_read"(%arg0, %arg2) {staticIndex = 3 : ui32, vector_width = 4 : i32} : (memref<?x4xf32>, index) -> f32
      %7 = "lo_spn.gaussian"(%3) {mean = 1.100000e-01 : f64, stddev = 1.000000e+00 : f64, supportMarginal = false, vector_width = 4 : i32} : (f32) -> !lo_spn.log<f64>
      %8 = "lo_spn.gaussian"(%4) {mean = 1.200000e-01 : f64, stddev = 7.500000e-01 : f64, supportMarginal = false, vector_width = 4 : i32} : (f32) -> !lo_spn.log<f64>
      %9 = "lo_spn.gaussian"(%5) {mean = 1.300000e-01 : f64, stddev = 5.000000e-01 : f64, supportMarginal = false, vector_width = 4 : i32} : (f32) -> !lo_spn.log<f64>
      %10 = "lo_spn.gaussian"(%6) {mean = 1.400000e-01 : f64, stddev = 2.500000e-01 : f64, supportMarginal = false, vector_width = 4 : i32} : (f32) -> !lo_spn.log<f64>
      %11 = "lo_spn.constant"() {type = !lo_spn.log<f64>, value = -1.3862943611198906 : f64, vector_width = 4 : i32} : () -> !lo_spn.log<f64>
      %12 = "lo_spn.mul"(%7, %11) {vector_width = 4 : i32} : (!lo_spn.log<f64>, !lo_spn.log<f64>) -> !lo_spn.log<f64>
      %13 = "lo_spn.mul"(%8, %11) {vector_width = 4 : i32} : (!lo_spn.log<f64>, !lo_spn.log<f64>) -> !lo_spn.log<f64>
      %14 = "lo_spn.add"(%12, %13) {vector_width = 4 : i32} : (!lo_spn.log<f64>, !lo_spn.log<f64>) -> !lo_spn.log<f64>
      %15 = "lo_spn.mul"(%9, %11) {vector_width = 4 : i32} : (!lo_spn.log<f64>, !lo_spn.log<f64>) -> !lo_spn.log<f64>
      %16 = "lo_spn.mul"(%10, %11) {vector_width = 4 : i32} : (!lo_spn.log<f64>, !lo_spn.log<f64>) -> !lo_spn.log<f64>
      %17 = "lo_spn.add"(%15, %16) {vector_width = 4 : i32} : (!lo_spn.log<f64>, !lo_spn.log<f64>) -> !lo_spn.log<f64>
      %18 = "lo_spn.add"(%14, %17) {vector_width = 4 : i32} : (!lo_spn.log<f64>, !lo_spn.log<f64>) -> !lo_spn.log<f64>
      %19 = "lo_spn.strip_log"(%18) {target = f64, vector_width = 4 : i32} : (!lo_spn.log<f64>) -> f64
      "lo_spn.batch_write"(%arg1, %arg2, %19) {vector_width = 4 : i32, transposed = true} : (memref<1x?xf64>, index, f64) -> ()
    }
    %c1 = constant 1 : index
    scf.for %arg2 = %2 to %0 step %c1 {
      %3 = "lo_spn.batch_read"(%arg0, %arg2) {staticIndex = 0 : ui32} : (memref<?x4xf32>, index) -> f32
      %4 = "lo_spn.batch_read"(%arg0, %arg2) {staticIndex = 1 : ui32} : (memref<?x4xf32>, index) -> f32
      %5 = "lo_spn.batch_read"(%arg0, %arg2) {staticIndex = 2 : ui32} : (memref<?x4xf32>, index) -> f32
      %6 = "lo_spn.batch_read"(%arg0, %arg2) {staticIndex = 3 : ui32} : (memref<?x4xf32>, index) -> f32
      %7 = "lo_spn.gaussian"(%3) {mean = 1.100000e-01 : f64, stddev = 1.000000e+00 : f64, supportMarginal = false} : (f32) -> !lo_spn.log<f64>
      %8 = "lo_spn.gaussian"(%4) {mean = 1.200000e-01 : f64, stddev = 7.500000e-01 : f64, supportMarginal = false} : (f32) -> !lo_spn.log<f64>
      %9 = "lo_spn.gaussian"(%5) {mean = 1.300000e-01 : f64, stddev = 5.000000e-01 : f64, supportMarginal = false} : (f32) -> !lo_spn.log<f64>
      %10 = "lo_spn.gaussian"(%6) {mean = 1.400000e-01 : f64, stddev = 2.500000e-01 : f64, supportMarginal = false} : (f32) -> !lo_spn.log<f64>
      %11 = "lo_spn.constant"() {type = !lo_spn.log<f64>, value = -1.3862943611198906 : f64} : () -> !lo_spn.log<f64>
      %12 = "lo_spn.mul"(%7, %11) : (!lo_spn.log<f64>, !lo_spn.log<f64>) -> !lo_spn.log<f64>
      %13 = "lo_spn.mul"(%8, %11) : (!lo_spn.log<f64>, !lo_spn.log<f64>) -> !lo_spn.log<f64>
      %14 = "lo_spn.add"(%12, %13) : (!lo_spn.log<f64>, !lo_spn.log<f64>) -> !lo_spn.log<f64>
      %15 = "lo_spn.mul"(%9, %11) : (!lo_spn.log<f64>, !lo_spn.log<f64>) -> !lo_spn.log<f64>
      %16 = "lo_spn.mul"(%10, %11) : (!lo_spn.log<f64>, !lo_spn.log<f64>) -> !lo_spn.log<f64>
      %17 = "lo_spn.add"(%15, %16) : (!lo_spn.log<f64>, !lo_spn.log<f64>) -> !lo_spn.log<f64>
      %18 = "lo_spn.add"(%14, %17) : (!lo_spn.log<f64>, !lo_spn.log<f64>) -> !lo_spn.log<f64>
      %19 = "lo_spn.strip_log"(%18) {target = f64} : (!lo_spn.log<f64>) -> f64
      "lo_spn.batch_write"(%arg1, %arg2, %19) {transposed = true} : (memref<1x?xf64>, index, f64) -> ()
    }
    return
  }
  func @spn_vector(%arg0: memref<?x4xf32>, %arg1: memref<1x?xf64>) {
    call @vec_task_0(%arg0, %arg1) : (memref<?x4xf32>, memref<1x?xf64>) -> ()
    "lo_spn.return"() : () -> ()
  }
}

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py


// CHECK-LABEL:   func @vec_task_0(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<?x4xf32>,
// CHECK-SAME:                     %[[VAL_1:.*]]: memref<1x?xf64>) {
// CHECK:           %[[VAL_2:.*]] = constant 0 : index
// CHECK:           %[[VAL_3:.*]] = memref.dim %[[VAL_0]], %[[VAL_2]] : memref<?x4xf32>
// CHECK:           %[[VAL_4:.*]] = constant 8 : index
// CHECK:           %[[VAL_5:.*]] = remi_unsigned %[[VAL_3]], %[[VAL_4]] : index
// CHECK:           %[[VAL_6:.*]] = subi %[[VAL_3]], %[[VAL_5]] : index
// CHECK:           %[[VAL_7:.*]] = constant 0 : index
// CHECK:           %[[VAL_8:.*]] = constant 8 : index
// CHECK:           scf.for %[[VAL_9:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_8]] {
// CHECK:             %[[VAL_10:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_11:.*]] = vector.broadcast %[[VAL_10]] : i64 to vector<4xi64>
// CHECK:             %[[VAL_12:.*]] = constant dense<[0, 4, 8, 12]> : vector<4xi64>
// CHECK:             %[[VAL_13:.*]] = constant dense<4> : vector<4xi64>
// CHECK:             %[[VAL_14:.*]] = muli %[[VAL_11]], %[[VAL_13]] : vector<4xi64>
// CHECK:             %[[VAL_15:.*]] = addi %[[VAL_14]], %[[VAL_12]] : vector<4xi64>
// CHECK:             %[[VAL_16:.*]] = constant dense<0.000000e+00> : vector<4xf32>
// CHECK:             %[[VAL_17:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_18:.*]] = constant 0 : index
// CHECK:             %[[VAL_19:.*]] = memref.dim %[[VAL_0]], %[[VAL_18]] : memref<?x4xf32>
// CHECK:             %[[VAL_20:.*]] = constant 4 : index
// CHECK:             %[[VAL_21:.*]] = muli %[[VAL_19]], %[[VAL_20]] : index
// CHECK:             %[[VAL_22:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_21]]], strides: [1] : memref<?x4xf32> to memref<?xf32>
// CHECK:             %[[VAL_23:.*]] = constant 0 : index
// CHECK:             %[[VAL_24:.*]] = vector.gather %[[VAL_22]]{{\[}}%[[VAL_23]]] {{\[}}%[[VAL_15]]], %[[VAL_17]], %[[VAL_16]] : memref<?xf32>, vector<4xi64>, vector<4xi1>, vector<4xf32> into vector<4xf32>
// CHECK:             %[[VAL_25:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_26:.*]] = vector.broadcast %[[VAL_25]] : i64 to vector<4xi64>
// CHECK:             %[[VAL_27:.*]] = constant dense<[1, 5, 9, 13]> : vector<4xi64>
// CHECK:             %[[VAL_28:.*]] = constant dense<4> : vector<4xi64>
// CHECK:             %[[VAL_29:.*]] = muli %[[VAL_26]], %[[VAL_28]] : vector<4xi64>
// CHECK:             %[[VAL_30:.*]] = addi %[[VAL_29]], %[[VAL_27]] : vector<4xi64>
// CHECK:             %[[VAL_31:.*]] = constant dense<0.000000e+00> : vector<4xf32>
// CHECK:             %[[VAL_32:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_33:.*]] = constant 0 : index
// CHECK:             %[[VAL_34:.*]] = memref.dim %[[VAL_0]], %[[VAL_33]] : memref<?x4xf32>
// CHECK:             %[[VAL_35:.*]] = constant 4 : index
// CHECK:             %[[VAL_36:.*]] = muli %[[VAL_34]], %[[VAL_35]] : index
// CHECK:             %[[VAL_37:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_36]]], strides: [1] : memref<?x4xf32> to memref<?xf32>
// CHECK:             %[[VAL_38:.*]] = constant 0 : index
// CHECK:             %[[VAL_39:.*]] = vector.gather %[[VAL_37]]{{\[}}%[[VAL_38]]] {{\[}}%[[VAL_30]]], %[[VAL_32]], %[[VAL_31]] : memref<?xf32>, vector<4xi64>, vector<4xi1>, vector<4xf32> into vector<4xf32>
// CHECK:             %[[VAL_40:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_41:.*]] = vector.broadcast %[[VAL_40]] : i64 to vector<4xi64>
// CHECK:             %[[VAL_42:.*]] = constant dense<[2, 6, 10, 14]> : vector<4xi64>
// CHECK:             %[[VAL_43:.*]] = constant dense<4> : vector<4xi64>
// CHECK:             %[[VAL_44:.*]] = muli %[[VAL_41]], %[[VAL_43]] : vector<4xi64>
// CHECK:             %[[VAL_45:.*]] = addi %[[VAL_44]], %[[VAL_42]] : vector<4xi64>
// CHECK:             %[[VAL_46:.*]] = constant dense<0.000000e+00> : vector<4xf32>
// CHECK:             %[[VAL_47:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_48:.*]] = constant 0 : index
// CHECK:             %[[VAL_49:.*]] = memref.dim %[[VAL_0]], %[[VAL_48]] : memref<?x4xf32>
// CHECK:             %[[VAL_50:.*]] = constant 4 : index
// CHECK:             %[[VAL_51:.*]] = muli %[[VAL_49]], %[[VAL_50]] : index
// CHECK:             %[[VAL_52:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_51]]], strides: [1] : memref<?x4xf32> to memref<?xf32>
// CHECK:             %[[VAL_53:.*]] = constant 0 : index
// CHECK:             %[[VAL_54:.*]] = vector.gather %[[VAL_52]]{{\[}}%[[VAL_53]]] {{\[}}%[[VAL_45]]], %[[VAL_47]], %[[VAL_46]] : memref<?xf32>, vector<4xi64>, vector<4xi1>, vector<4xf32> into vector<4xf32>
// CHECK:             %[[VAL_55:.*]] = index_cast %[[VAL_9]] : index to i64
// CHECK:             %[[VAL_56:.*]] = vector.broadcast %[[VAL_55]] : i64 to vector<4xi64>
// CHECK:             %[[VAL_57:.*]] = constant dense<[3, 7, 11, 15]> : vector<4xi64>
// CHECK:             %[[VAL_58:.*]] = constant dense<4> : vector<4xi64>
// CHECK:             %[[VAL_59:.*]] = muli %[[VAL_56]], %[[VAL_58]] : vector<4xi64>
// CHECK:             %[[VAL_60:.*]] = addi %[[VAL_59]], %[[VAL_57]] : vector<4xi64>
// CHECK:             %[[VAL_61:.*]] = constant dense<0.000000e+00> : vector<4xf32>
// CHECK:             %[[VAL_62:.*]] = constant dense<true> : vector<4xi1>
// CHECK:             %[[VAL_63:.*]] = constant 0 : index
// CHECK:             %[[VAL_64:.*]] = memref.dim %[[VAL_0]], %[[VAL_63]] : memref<?x4xf32>
// CHECK:             %[[VAL_65:.*]] = constant 4 : index
// CHECK:             %[[VAL_66:.*]] = muli %[[VAL_64]], %[[VAL_65]] : index
// CHECK:             %[[VAL_67:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: {{\[}}%[[VAL_66]]], strides: [1] : memref<?x4xf32> to memref<?xf32>
// CHECK:             %[[VAL_68:.*]] = constant 0 : index
// CHECK:             %[[VAL_69:.*]] = vector.gather %[[VAL_67]]{{\[}}%[[VAL_68]]] {{\[}}%[[VAL_60]]], %[[VAL_62]], %[[VAL_61]] : memref<?xf32>, vector<4xi64>, vector<4xi1>, vector<4xf32> into vector<4xf32>
// CHECK:             %[[VAL_70:.*]] = fpext %[[VAL_24]] : vector<4xf32> to vector<4xf64>
// CHECK:             %[[VAL_71:.*]] = constant dense<-5.000000e-01> : vector<4xf64>
// CHECK:             %[[VAL_72:.*]] = constant dense<-0.91893853320467266> : vector<4xf64>
// CHECK:             %[[VAL_73:.*]] = constant dense<1.100000e-01> : vector<4xf64>
// CHECK:             %[[VAL_74:.*]] = subf %[[VAL_70]], %[[VAL_73]] : vector<4xf64>
// CHECK:             %[[VAL_75:.*]] = mulf %[[VAL_74]], %[[VAL_74]] : vector<4xf64>
// CHECK:             %[[VAL_76:.*]] = mulf %[[VAL_75]], %[[VAL_71]] : vector<4xf64>
// CHECK:             %[[VAL_77:.*]] = addf %[[VAL_72]], %[[VAL_76]] : vector<4xf64>
// CHECK:             %[[VAL_78:.*]] = fpext %[[VAL_39]] : vector<4xf32> to vector<4xf64>
// CHECK:             %[[VAL_79:.*]] = constant dense<-0.88888888888888883> : vector<4xf64>
// CHECK:             %[[VAL_80:.*]] = constant dense<-0.63125646075289177> : vector<4xf64>
// CHECK:             %[[VAL_81:.*]] = constant dense<1.200000e-01> : vector<4xf64>
// CHECK:             %[[VAL_82:.*]] = subf %[[VAL_78]], %[[VAL_81]] : vector<4xf64>
// CHECK:             %[[VAL_83:.*]] = mulf %[[VAL_82]], %[[VAL_82]] : vector<4xf64>
// CHECK:             %[[VAL_84:.*]] = mulf %[[VAL_83]], %[[VAL_79]] : vector<4xf64>
// CHECK:             %[[VAL_85:.*]] = addf %[[VAL_80]], %[[VAL_84]] : vector<4xf64>
// CHECK:             %[[VAL_86:.*]] = fpext %[[VAL_54]] : vector<4xf32> to vector<4xf64>
// CHECK:             %[[VAL_87:.*]] = constant dense<-2.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_88:.*]] = constant dense<-0.22579135264472738> : vector<4xf64>
// CHECK:             %[[VAL_89:.*]] = constant dense<1.300000e-01> : vector<4xf64>
// CHECK:             %[[VAL_90:.*]] = subf %[[VAL_86]], %[[VAL_89]] : vector<4xf64>
// CHECK:             %[[VAL_91:.*]] = mulf %[[VAL_90]], %[[VAL_90]] : vector<4xf64>
// CHECK:             %[[VAL_92:.*]] = mulf %[[VAL_91]], %[[VAL_87]] : vector<4xf64>
// CHECK:             %[[VAL_93:.*]] = addf %[[VAL_88]], %[[VAL_92]] : vector<4xf64>
// CHECK:             %[[VAL_94:.*]] = fpext %[[VAL_69]] : vector<4xf32> to vector<4xf64>
// CHECK:             %[[VAL_95:.*]] = constant dense<-8.000000e+00> : vector<4xf64>
// CHECK:             %[[VAL_96:.*]] = constant dense<0.4673558279152179> : vector<4xf64>
// CHECK:             %[[VAL_97:.*]] = constant dense<1.400000e-01> : vector<4xf64>
// CHECK:             %[[VAL_98:.*]] = subf %[[VAL_94]], %[[VAL_97]] : vector<4xf64>
// CHECK:             %[[VAL_99:.*]] = mulf %[[VAL_98]], %[[VAL_98]] : vector<4xf64>
// CHECK:             %[[VAL_100:.*]] = mulf %[[VAL_99]], %[[VAL_95]] : vector<4xf64>
// CHECK:             %[[VAL_101:.*]] = addf %[[VAL_96]], %[[VAL_100]] : vector<4xf64>
// CHECK:             %[[VAL_102:.*]] = constant dense<-1.3862943611198906> : vector<4xf64>
// CHECK:             %[[VAL_103:.*]] = addf %[[VAL_77]], %[[VAL_102]] : vector<4xf64>
// CHECK:             %[[VAL_104:.*]] = addf %[[VAL_85]], %[[VAL_102]] : vector<4xf64>
// CHECK:             %[[VAL_105:.*]] = cmpf ogt, %[[VAL_103]], %[[VAL_104]] : vector<4xf64>
// CHECK:             %[[VAL_106:.*]] = select %[[VAL_105]], %[[VAL_103]], %[[VAL_104]] : vector<4xi1>, vector<4xf64>
// CHECK:             %[[VAL_107:.*]] = select %[[VAL_105]], %[[VAL_104]], %[[VAL_103]] : vector<4xi1>, vector<4xf64>
// CHECK:             %[[VAL_108:.*]] = subf %[[VAL_107]], %[[VAL_106]] : vector<4xf64>
// CHECK:             %[[VAL_109:.*]] = math.exp %[[VAL_108]] : vector<4xf64>
// CHECK:             %[[VAL_110:.*]] = math.log1p %[[VAL_109]] : vector<4xf64>
// CHECK:             %[[VAL_111:.*]] = addf %[[VAL_106]], %[[VAL_110]] : vector<4xf64>
// CHECK:             %[[VAL_112:.*]] = addf %[[VAL_93]], %[[VAL_102]] : vector<4xf64>
// CHECK:             %[[VAL_113:.*]] = addf %[[VAL_101]], %[[VAL_102]] : vector<4xf64>
// CHECK:             %[[VAL_114:.*]] = cmpf ogt, %[[VAL_112]], %[[VAL_113]] : vector<4xf64>
// CHECK:             %[[VAL_115:.*]] = select %[[VAL_114]], %[[VAL_112]], %[[VAL_113]] : vector<4xi1>, vector<4xf64>
// CHECK:             %[[VAL_116:.*]] = select %[[VAL_114]], %[[VAL_113]], %[[VAL_112]] : vector<4xi1>, vector<4xf64>
// CHECK:             %[[VAL_117:.*]] = subf %[[VAL_116]], %[[VAL_115]] : vector<4xf64>
// CHECK:             %[[VAL_118:.*]] = math.exp %[[VAL_117]] : vector<4xf64>
// CHECK:             %[[VAL_119:.*]] = math.log1p %[[VAL_118]] : vector<4xf64>
// CHECK:             %[[VAL_120:.*]] = addf %[[VAL_115]], %[[VAL_119]] : vector<4xf64>
// CHECK:             %[[VAL_121:.*]] = cmpf ogt, %[[VAL_111]], %[[VAL_120]] : vector<4xf64>
// CHECK:             %[[VAL_122:.*]] = select %[[VAL_121]], %[[VAL_111]], %[[VAL_120]] : vector<4xi1>, vector<4xf64>
// CHECK:             %[[VAL_123:.*]] = select %[[VAL_121]], %[[VAL_120]], %[[VAL_111]] : vector<4xi1>, vector<4xf64>
// CHECK:             %[[VAL_124:.*]] = subf %[[VAL_123]], %[[VAL_122]] : vector<4xf64>
// CHECK:             %[[VAL_125:.*]] = math.exp %[[VAL_124]] : vector<4xf64>
// CHECK:             %[[VAL_126:.*]] = math.log1p %[[VAL_125]] : vector<4xf64>
// CHECK:             %[[VAL_127:.*]] = addf %[[VAL_122]], %[[VAL_126]] : vector<4xf64>
// CHECK:             %[[VAL_128:.*]] = constant 0 : index
// CHECK:             vector.transfer_write %[[VAL_127]], %[[VAL_1]]{{\[}}%[[VAL_128]], %[[VAL_9]]] : vector<4xf64>, memref<1x?xf64>
// CHECK:           }
// CHECK:           %[[VAL_129:.*]] = constant 1 : index
// CHECK:           scf.for %[[VAL_130:.*]] = %[[VAL_6]] to %[[VAL_3]] step %[[VAL_129]] {
// CHECK:             %[[VAL_131:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_130]]) {staticIndex = 0 : ui32} : (memref<?x4xf32>, index) -> f32
// CHECK:             %[[VAL_132:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_130]]) {staticIndex = 1 : ui32} : (memref<?x4xf32>, index) -> f32
// CHECK:             %[[VAL_133:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_130]]) {staticIndex = 2 : ui32} : (memref<?x4xf32>, index) -> f32
// CHECK:             %[[VAL_134:.*]] = "lo_spn.batch_read"(%[[VAL_0]], %[[VAL_130]]) {staticIndex = 3 : ui32} : (memref<?x4xf32>, index) -> f32
// CHECK:             %[[VAL_135:.*]] = "lo_spn.gaussian"(%[[VAL_131]]) {mean = 1.100000e-01 : f64, stddev = 1.000000e+00 : f64, supportMarginal = false} : (f32) -> !lo_spn.log<f64>
// CHECK:             %[[VAL_136:.*]] = "lo_spn.gaussian"(%[[VAL_132]]) {mean = 1.200000e-01 : f64, stddev = 7.500000e-01 : f64, supportMarginal = false} : (f32) -> !lo_spn.log<f64>
// CHECK:             %[[VAL_137:.*]] = "lo_spn.gaussian"(%[[VAL_133]]) {mean = 1.300000e-01 : f64, stddev = 5.000000e-01 : f64, supportMarginal = false} : (f32) -> !lo_spn.log<f64>
// CHECK:             %[[VAL_138:.*]] = "lo_spn.gaussian"(%[[VAL_134]]) {mean = 1.400000e-01 : f64, stddev = 2.500000e-01 : f64, supportMarginal = false} : (f32) -> !lo_spn.log<f64>
// CHECK:             %[[VAL_139:.*]] = "lo_spn.constant"() {type = !lo_spn.log<f64>, value = -1.3862943611198906 : f64} : () -> !lo_spn.log<f64>
// CHECK:             %[[VAL_140:.*]] = "lo_spn.mul"(%[[VAL_135]], %[[VAL_139]]) : (!lo_spn.log<f64>, !lo_spn.log<f64>) -> !lo_spn.log<f64>
// CHECK:             %[[VAL_141:.*]] = "lo_spn.mul"(%[[VAL_136]], %[[VAL_139]]) : (!lo_spn.log<f64>, !lo_spn.log<f64>) -> !lo_spn.log<f64>
// CHECK:             %[[VAL_142:.*]] = "lo_spn.add"(%[[VAL_140]], %[[VAL_141]]) : (!lo_spn.log<f64>, !lo_spn.log<f64>) -> !lo_spn.log<f64>
// CHECK:             %[[VAL_143:.*]] = "lo_spn.mul"(%[[VAL_137]], %[[VAL_139]]) : (!lo_spn.log<f64>, !lo_spn.log<f64>) -> !lo_spn.log<f64>
// CHECK:             %[[VAL_144:.*]] = "lo_spn.mul"(%[[VAL_138]], %[[VAL_139]]) : (!lo_spn.log<f64>, !lo_spn.log<f64>) -> !lo_spn.log<f64>
// CHECK:             %[[VAL_145:.*]] = "lo_spn.add"(%[[VAL_143]], %[[VAL_144]]) : (!lo_spn.log<f64>, !lo_spn.log<f64>) -> !lo_spn.log<f64>
// CHECK:             %[[VAL_146:.*]] = "lo_spn.add"(%[[VAL_142]], %[[VAL_145]]) : (!lo_spn.log<f64>, !lo_spn.log<f64>) -> !lo_spn.log<f64>
// CHECK:             %[[VAL_147:.*]] = "lo_spn.strip_log"(%[[VAL_146]]) {target = f64} : (!lo_spn.log<f64>) -> f64
// CHECK:             "lo_spn.batch_write"(%[[VAL_1]], %[[VAL_130]], %[[VAL_147]]) {transposed = true} : (memref<1x?xf64>, index, f64) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func @spn_vector(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<?x4xf32>,
// CHECK-SAME:                     %[[VAL_1:.*]]: memref<1x?xf64>) {
// CHECK:           call @vec_task_0(%[[VAL_0]], %[[VAL_1]]) : (memref<?x4xf32>, memref<1x?xf64>) -> ()
// CHECK:           "lo_spn.return"() : () -> ()
// CHECK:         }
