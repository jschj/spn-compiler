#pragma once

#include "firp.hpp"


namespace mlir::spn::fpga::fixedpoint {

uint64_t doubleToFixed(uint32_t intBits, uint32_t fracBits, double value);
double fixedToDouble(uint32_t intBits, uint32_t fracBits, uint64_t value);

class FixedAdd : public firp::Module<FixedAdd> {
  uint32_t intBits, fracBits;
public:
  FixedAdd(uint32_t intBits, uint32_t fracBits):
    firp::Module<FixedAdd>(
      "FixedAdd",
      {
        firp::Port("a", true, firp::uintType(intBits + fracBits)),
        firp::Port("b", true, firp::uintType(intBits + fracBits)),
        firp::Port("c", false, firp::uintType(intBits + fracBits))
      },
      intBits, fracBits
    ) { build(); }

  void body();

  uint32_t getDelay() const {
    return 1;
  }
};

class FixedMul : public firp::Module<FixedMul> {
  uint32_t intBits, fracBits;
public:
  FixedMul(uint32_t intBits, uint32_t fracBits):
    firp::Module<FixedMul>(
      "FixedMul",
      {
        firp::Port("a", true, firp::uintType(intBits + fracBits)),
        firp::Port("b", true, firp::uintType(intBits + fracBits)),
        firp::Port("c", false, firp::uintType(intBits + fracBits))
      },
      intBits, fracBits
    ) { build(); }

  void body();

  uint32_t getDelay() const {
    return 1;
  }
};

class FixedLog : public firp::Module<FixedLog> {
  uint32_t intBits, fracBits;
public:
  FixedLog(uint32_t intBits, uint32_t fracBits):
    firp::Module<FixedLog>(
      "FixedLog",
      {
        firp::Port("a", true, firp::uintType(intBits + fracBits)),
        firp::Port("c", false, firp::uintType(intBits + fracBits))
      },
      intBits, fracBits
    ) { build(); }

  void body() {
    // is a no-op
    io("c") <<= io("a");
  }

  uint32_t getDelay() const {
    return 0;
  }
};

class FixedHistogram : public firp::Module<FixedHistogram> {
  uint32_t intBits, fracBits;
  std::vector<double> probabilities;
public:
  FixedHistogram(uint32_t intBits, uint32_t fracBits, uint32_t instanceId, const std::vector<double>& probabilities):
    firp::Module<FixedHistogram>(
      "FixedHistogram",
      {
        firp::Port("index", true, firp::uintType(firp::clog2(probabilities.size()))),
        firp::Port("prob", false, firp::uintType(intBits + fracBits))
      },
      intBits, fracBits, instanceId
    ),
    intBits(intBits), fracBits(fracBits),
    probabilities(probabilities)
  { build(); }

  void body();

  uint32_t getDelay() const {
    return 1;
  }
};

}