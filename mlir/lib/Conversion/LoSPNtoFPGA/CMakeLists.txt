message("FIRRTLPP include path: " ${FIRRTLPP_INCLUDE_DIRS})

add_mlir_conversion_library(MLIRLoSPNtoFPGATransforms
    LoSPNtoFPGAPass.cpp
    conversion.cpp
    DelayInsertionPass.cpp
    ufloat.cpp
    fixedpoint.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/mlir/include/Conversion
    ${PROJECT_SOURCE_DIR}/lo2hw

    LINK_LIBS PUBLIC
    MLIRLoSPN
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRMathDialect

    CIRCTHW
    CIRCTHWArith
    CIRCTComb
    CIRCTScheduling
    CIRCTSeq
    CIRCTFIRRTL
)

target_link_libraries(MLIRLoSPNtoFPGATransforms PRIVATE firp)

#message("adding include  dir " ${PROJECT_SOURCE_DIR}/lo2hw)

target_include_directories(MLIRLoSPNtoFPGATransforms PRIVATE ${PROJECT_SOURCE_DIR}/common/include ${FIRRTLPP_INCLUDE_DIRS})
#target_include_directories(MLIRLoSPNtoFPGATransforms PRIVATE ${PROJECT_SOURCE_DIR}/lo2hw)

#add_executable(ufloat ufloat.cpp)
#target_include_directories(ufloat PRIVATE
#  ${LLVM_INCLUDE_DIRS}
#  ${MLIR_INCLUDE_DIRS}
#  ${FIRRTLPP_INCLUDE_DIRS}
#)
#target_link_libraries(ufloat PRIVATE ${LIBS} CIRCTHW CIRCTSeq CIRCTFIRRTL firp)

add_executable(fixedpoint fixedpoint.cpp)
target_include_directories(fixedpoint PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${FIRRTLPP_INCLUDE_DIRS}
)
target_link_libraries(fixedpoint PRIVATE ${LIBS} CIRCTHW CIRCTSeq CIRCTFIRRTL firp)

#add_compile_definitions(TEST_KERNEL_DIR="${CMAKE_CURRENT_BINARY_DIR}")

#set(SPNC_DRIVER_BINARY_PATH "${CMAKE_CURRENT_BINARY_DIR}/driver" CACHE INTERNAL "")
