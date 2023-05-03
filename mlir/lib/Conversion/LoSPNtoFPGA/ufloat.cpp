#include "LoSPNtoFPGA/ufloat.hpp"


namespace mlir::spn::fpga::ufloat {

using namespace firp;

circt::firrtl::BundleType ufloatType(uint32_t ew, uint32_t mw) {
  return bundleType({
    {"e", false, uintType(ew)},
    {"m", false, uintType(mw)}
  });
}

FValue uintToUFloat(FValue input, uint32_t ew, uint32_t mw) {
  auto result = Wire(ufloatType(ew, mw));
  result("e") <<= input(ew + mw - 1, mw);
  result("m") <<= input(mw - 1, 0);
  return result;
}

FValue ufloatToUInt(FValue input, uint32_t ew, uint32_t mw) {
  return cat({input("e"), input("m")});
}

uint64_t doubleToUFloat(double d, uint32_t ew, uint32_t mw) {
  assert(ew + mw <= 64);

  uint64_t l = *reinterpret_cast<const uint64_t *>(&d);
  uint64_t m = l & 0xFFFFFFFFFFFFFL;
  uint64_t e = l >> 52;
  uint64_t offset_double = (1L << (11 - 1)) - 1;
  uint64_t offset_ufloat = (1L << (ew - 1)) - 1;
  uint64_t offset_delta = offset_ufloat - offset_double;
  uint64_t e_new = e + offset_delta;
  uint64_t fewer_mantissa_bits = 52 - mw;
  uint64_t m_new = m >> fewer_mantissa_bits;
  uint64_t result = e_new << mw | m_new;

  if (l == 0 || e_new < 0) {
    return 0L;
  } else {
    assert(clog2(result) <= ew + mw);
    return result;
  }
}

double ufloatToDouble(uint64_t u, uint32_t ew, uint32_t mw) {
  uint64_t expMask = (1ul << ew) - 1;
  uint64_t manMask = (1ul << mw) - 1;

  uint64_t e = (u >> mw) & expMask;
  uint64_t m = u & manMask;

  uint64_t offset_double = (1L << (11 - 1)) - 1;
  uint64_t offset_input = (1L << (ew - 1)) - 1;
  uint64_t offset_delta = offset_double - offset_input;

  uint64_t extra_mantissa_bits = 53 - mw;
  // TODO: Not sure about this.
  uint64_t new_mantissa = m << (extra_mantissa_bits - 1);
  uint64_t new_exponent_nonzero = e + offset_delta;
  uint64_t inputs_zero = e == 0 && m == 0;

  if (inputs_zero) {
    return 0.0;
  } else {
    uint64_t double_bits = (new_mantissa | (new_exponent_nonzero << 52));
    return *reinterpret_cast<const double *>(&double_bits);
  }
}

}


namespace mlir::spn::fpga::ufloat::adder {

using namespace firp;

class PipelinedAdder : public Module<PipelinedAdder> {
  uint32_t width, chunkSize;
  bool sub;
public:
  PipelinedAdder(uint32_t width, uint32_t chunkSize):
    Module<PipelinedAdder>(
      "PipelineAdder",
      {
        Port("a", true, uintType(width)),
        Port("b", true, uintType(width)),
        Port("r", false, uintType(width + 1))
      },
      width, chunkSize
    ), width(width), chunkSize(chunkSize) {}

  void body() {
    uint32_t internalWidth = (width % chunkSize == 0) ? width : width + chunkSize - (width % chunkSize);
    uint32_t stages = internalWidth / chunkSize;

    auto aPad = (internalWidth > width) ? cat({cons(0, uintType(internalWidth - width)), io("a")}) : io("a");
    auto bPad = (internalWidth > width) ? cat({cons(0, uintType(internalWidth - width)), io("b")}) : io("b");

    std::vector<FValue> a, b, aDelayed, bDelayed,
      carrySignals{cons(0, uintType(1))}, resultSignals;

    for (uint32_t x = 0; x < stages; ++x) {
      // TODO: Check index bounds
      a.push_back(aPad((x + 1) * chunkSize - 1, x * chunkSize));
      b.push_back(bPad((x + 1) * chunkSize - 1, x * chunkSize));
    }

    for (uint32_t i = 0; i < stages; ++i) {
      aDelayed.push_back(shiftRegister(a[i], i));
      bDelayed.push_back(shiftRegister(b[i], i));
    }

    for (uint32_t i = 0; i < stages; ++i) {
      auto aCurrent = aDelayed[i];
      auto bCurrent = bDelayed[i];
      auto carryIn = carrySignals.back();

      auto rSum = aCurrent + bCurrent + carryIn;
      llvm::outs() << "bit counts: " << aCurrent.bitCount() << " " << bCurrent.bitCount() << " " << carryIn.bitCount() <<
        " " << rSum.bitCount() << "\n";
      auto rCurrent = rSum(rSum.bitCount() - 2, 0);
      auto carryOut = rSum(rSum.bitCount() - 1);
      //auto (rCurrent, carryOut) = if (sub) -&|(aCurrent, bCurrent, carryIn) else +&|(aCurrent, bCurrent, carryIn);

      carrySignals.push_back(carryOut);
      resultSignals.push_back(rCurrent);
    }

    std::vector<FValue> flushRes;

    for (int32_t i = stages - 1; i >= 0; --i)
      flushRes.push_back(shiftRegister(resultSignals[stages - 1 - i], i));

    if (chunkSize == internalWidth) {
      io("r") <<= cat({carrySignals.back(), flushRes.front()});
    } else {
      std::vector<FValue> reversed{carrySignals.back()};
      std::reverse_copy(flushRes.cbegin(), flushRes.cend(), std::back_inserter(reversed));
      io("r") <<= cat(reversed);
    }

  }

  static uint32_t getDelay(uint32_t width, uint32_t chunkSize) {
    uint32_t internalWidth = (width % chunkSize == 0) ? width : width + chunkSize - (width % chunkSize);
    uint32_t stages = internalWidth / chunkSize;
    return stages;
  }
};

class CompareComb : public Module<CompareComb> {
  uint32_t width;
public:
  CompareComb(uint32_t width):
    Module<CompareComb>(
      "CompareComb",
      {
        Port("op1", true, uintType(width)),
        Port("op2", true, uintType(width)),
        Port("gte", false, bitType())
      },
      width
    ),
    width(width) {}

  void body() {
    io("gte") <<= io("op1") >= io("op2");
  }
};

class Swap : public Module<Swap> {
  uint32_t exponentWidth, mantissaWidth;
public:
  Swap(uint32_t exponentWidth, uint32_t mantissaWidth):
    Module<Swap>(
      "Swap",
      {
        Port("in_op1", true, ufloatType(exponentWidth, mantissaWidth)),
        Port("in_op2", true, ufloatType(exponentWidth, mantissaWidth)),
        Port("out_op1", false, ufloatType(exponentWidth, mantissaWidth)),
        Port("out_op2", false, ufloatType(exponentWidth, mantissaWidth))
      },
      exponentWidth, mantissaWidth
    ),
    exponentWidth(exponentWidth), mantissaWidth(mantissaWidth) {}

  void body() {
    CompareComb compare(exponentWidth);
    compare.io("op1") <<= io("in_op1");
    compare.io("op2") <<= io("in_op2");

    io("out_op1") <<= regNext(mux(compare.io("gte"), io("in_op1"), io("in_op2")));
    io("out_op2") <<= regNext(mux(compare.io("gte"), io("in_op1"), io("in_op2")));
  }
};

class SubtractStage : public Module<SubtractStage> {
  uint32_t width;
public:
  SubtractStage(uint32_t width):
    Module<SubtractStage>(
      "SubtractStage",
      {
        Port("minuend", true, uintType(width)),
        Port("subtrahend", true, uintType(width)),
        Port("difference", false, uintType(width)),
        Port("minuend_is_zero", false, bitType())
      },
      width
    ), width(width) {}

  void body() {
    auto difference = regNext(io("minuend") - io("subtrahend"));
    io("difference") <<= difference;
    io("minuend_is_zero") <<= regNext(io("minuend") == cons(0, uintType(width)));
  }
};

class ShiftStage : public Module<ShiftStage> {
  uint32_t valueSize, shiftAmountSize;
public:
  ShiftStage(uint32_t valueSize, uint32_t shiftAmountSize):
    Module<ShiftStage>(
      "ShiftStage",
      {
        Port("value", true, uintType(valueSize)),
        Port("shamt", true, uintType(shiftAmountSize)),
        Port("out", false, uintType(valueSize + 1))
      },
      valueSize, shiftAmountSize
    ),
    valueSize(valueSize), shiftAmountSize(shiftAmountSize) {}

  void body() {
    auto shifted = regNext(cat({cons(1, bitType()), io("value")}) >> io("shamt"));
    io("out") <<= shifted;
  }
};

class MantissaShifterComb : public Module<MantissaShifterComb> {
  uint32_t mantissaWidth;
public:
  MantissaShifterComb(uint32_t mantissaWidth):
    Module<MantissaShifterComb>(
      "MantissaShifterComb",
      {
        Port("sum", true, uintType(mantissaWidth + 2)),
        Port("out", false, uintType(mantissaWidth)),
        Port("shifted", false, uintType(1))
      },
      mantissaWidth
    ), mantissaWidth(mantissaWidth) {}

  void body() {
    // TODO: Check if these relate to the correct msb/lsb
    auto shiftedSum = mux(
      io("sum")(mantissaWidth + 2 - 1),
      io("sum")(0),
      io("sum")(1, 0)
    );
    auto shifted = io("sum")(mantissaWidth + 2 - 1);
    io("out") <<= shiftedSum;
    io("shifted") <<= shifted;
  }
};

class ExponentAdder : public Module<ExponentAdder> {
public:
  ExponentAdder(uint32_t exponentWidth):
    Module<ExponentAdder>(
      "ExponentAdder",
      {
        Port("e_in", true, uintType(exponentWidth)),
        Port("add", true, bitType()),
        Port("inputs_are_zero", true, bitType()),
        Port("e_out", false, uintType(exponentWidth))
      },
      exponentWidth
    ) {}

  void body() {
    io("e_out") <<= regNext(
      mux(
        io("add") & ~io("inputs_are_zero"),
        io("e_in") + cons(1, uintType(1)),
        io("e_in")
      ),
      "out"
    );
  }
};

FPAdd::FPAdd(uint32_t exponentWidth, uint32_t mantissaWidth):
  Module<FPAdd>(
    "FPAdd",
    {
      Port("a", true, uintType(exponentWidth + mantissaWidth)),
      Port("b", true, uintType(exponentWidth + mantissaWidth)),
      Port("r", false, uintType(exponentWidth + mantissaWidth)) // ???
    },
    exponentWidth, mantissaWidth
  ),
  exponentWidth(exponentWidth), mantissaWidth(mantissaWidth) {}

void FPAdd::body() {
  auto op1_packed = regNext(io("a"));
  auto op2_packed = regNext(io("b"));

  // TODO
  auto op1 = uintToUFloat(op1_packed, exponentWidth, mantissaWidth);
  auto op2 = uintToUFloat(op2_packed, exponentWidth, mantissaWidth);

  // Stage 1: Swap
  auto swap = Swap(exponentWidth, mantissaWidth);
  swap.io("in_op1") <<= op1;
  swap.io("in_op2") <<= op2;

  // Stage 2: Subtract Mantissas
  auto subtractStage = SubtractStage(exponentWidth);
  subtractStage.io("minuend") <<= swap.io("out_op1")("e");
  subtractStage.io("subtrahend") <<= swap.io("out_op2")("e");
  auto difference_2 = subtractStage.io("difference");
  auto m1_2 = regNext(swap.io("out_op1")("m"));
  auto m2_2 = regNext(swap.io("out_op2")("m"));
  auto e1_2 = regNext(swap.io("out_op1")("e"));
  auto e2_2 = regNext(swap.io("out_op2")("e"));

  // Stage 3: Shift smaller Mantissa
  auto shiftStage = ShiftStage(exponentWidth, mantissaWidth); // Stage 3
  shiftStage.io("value") <<= m2_2;
  shiftStage.io("shamt") <<= difference_2;
  auto m1_3 = regNext(m1_2);
  auto m2_3 = shiftStage.io("out");
  auto e1_3 = regNext(e1_2);
  auto minuend_is_zero_3 = regNext(subtractStage.io("minuend_is_zero"));

  // Stage 4: Add Mantissas
  auto pipelinedAdder = PipelinedAdder(mantissaWidth + 1, 32);
  auto a = cat({cons(1, bitType()), m1_3});
  pipelinedAdder.io("a") <<= a;
  pipelinedAdder.io("b") <<= m2_3;
  auto mantissaSum = pipelinedAdder.io("r");
  auto pipelinedAdderDelay = PipelinedAdder::getDelay(mantissaWidth + 1, 32);
  auto minuend_is_zero_4 = shiftRegister(minuend_is_zero_3, pipelinedAdderDelay);
  auto e1_4 = shiftRegister(e1_3, pipelinedAdderDelay);

  // Stage 5: Shift Sum if necessary and increment exponent
  auto mantissaShifterComb = MantissaShifterComb(mantissaWidth);
  mantissaShifterComb.io("sum") <<= mantissaSum;
  auto shifted = mantissaShifterComb.io("shifted");

  auto exponentAdder = ExponentAdder(exponentWidth);
  exponentAdder.io("e_in") <<= e1_4;
  exponentAdder.io("add") <<= shifted;
  exponentAdder.io("inputs_are_zero") <<= minuend_is_zero_4;
  auto m_6 = regNext(mantissaShifterComb.io("out"));

  // Stage 6: output Result
  auto out = Wire(ufloatType(mantissaWidth, exponentWidth));
  out("m") <<= m_6;
  out("e") <<= exponentAdder.io("e_out");
  auto output_together = ufloatToUInt(out, exponentWidth, mantissaWidth);
  io("r") <<= output_together;
}

}

/*
int main() {
  std::unique_ptr<mlir::MLIRContext> context = std::make_unique<mlir::MLIRContext>();

  assert(context->getOrLoadDialect<circt::hw::HWDialect>());
  assert(context->getOrLoadDialect<circt::seq::SeqDialect>());
  assert(context->getOrLoadDialect<circt::firrtl::FIRRTLDialect>());
  assert(context->getOrLoadDialect<circt::sv::SVDialect>());
  //assert(context->getOrLoadDialect<circt::esi::ESIDialect>());

  using namespace ::firp;
  using namespace ::circt::firrtl;
  using namespace ::mlir;
  using namespace mlir::spn::fpga::ufloat::adder;

  initFirpContext(context.get(), "FPAdd");

  {
    FPAdd add(8, 23);
    add.makeTop();
  }

  firpContext()->finish();
  firpContext()->dump();

  return 0;
}
 */