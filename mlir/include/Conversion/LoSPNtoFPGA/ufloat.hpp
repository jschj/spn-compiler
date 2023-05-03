#pragma once

#include "firp.hpp"


namespace mlir::spn::fpga::ufloat {

circt::firrtl::BundleType ufloatType(uint32_t ew, uint32_t mw);
firp::FValue uintToUFloat(firp::FValue input, uint32_t ew, uint32_t mw);
firp::FValue ufloatToUInt(firp::FValue input, uint32_t ew, uint32_t mw);

uint64_t doubleToUFloat(double d, uint32_t ew, uint32_t mw);
double ufloatToDouble(uint64_t u, uint32_t ew, uint32_t mw);

namespace adder {

class FPAdd : public firp::Module<FPAdd> {
  uint32_t exponentWidth, mantissaWidth;
public:
  FPAdd(uint32_t exponentWidth, uint32_t mantissaWidth);
  void body();
};

}

}

