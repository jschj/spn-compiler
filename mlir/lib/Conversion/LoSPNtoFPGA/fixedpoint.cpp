#include "LoSPNtoFPGA/fixedpoint.hpp"


namespace mlir::spn::fpga::fixedpoint {

using namespace firp;

/*
static FValue pipelinedAdd(FValue a, FValue b, uint32_t maxAdderSize) {
  // inspired by http://vlsigyan.com/pipeline-adder-verilog-code/

  uint32_t maxWidth = std::make(a.bitCount(), b.bitCount());

  // zero extend the value with the lesser bit count
  auto a = a.extend(maxWidth);
  auto b = b.extend(maxWidth);
  Wire c(uintType(maxWidth));
  FValue prevCarry = cons(0, uintType(1));

  for (uint32_t i = 0, j = 0; i < maxWidth; i += maxAdderSize, ++j) {
    uint32_t hi = std::min((i + 1) * maxAdderSize, maxWidth);
    uint32_t lo = i * maxAdderSize;
    uint32_t preDelay = j;
    uint32_t postDelay = (maxWidth / maxAdderSize) - j - 1;

    auto sum =
      shiftRegister(a(hi, lo), preDelay) +
      shiftRegister(b(hi, lo), preDelay) +
      prevCarry;

    c(hi, lo) <<= shiftRegister(sum(maxWidth - 1, 0), postDelay);
    prevCarry = regNext(sum(maxWidth));
  }

  return c;
}
 */

uint64_t doubleToFixed(uint32_t intBits, uint32_t fracBits, double value) {
  assert(fracBits < 64 && "too many fractional bits");
  assert(value >= 0.0 && "value must be positive");

  uint64_t scale = 1ul << fracBits;
  uint64_t c = std::round(value * scale);

  return c;
}

double fixedToDouble(uint32_t intBits, uint32_t fracBits, uint64_t value) {
  uint64_t intMask = (1ul << intBits) - 1;
  uint64_t fracMask = (1ul << fracBits) - 1;

  uint64_t intPart = (value >> intBits) & intMask;
  uint64_t fracPart = value & fracMask;
  uint64_t scale = 1ul << fracMask;

  return intPart + double(fracPart) / scale;
}

void FixedAdd::body() {
  // a/s + b/s = (a + b) / s
  io("c") <<= regNext(io("a") + io("b"), "c");
}

void FixedMul::body() {
  // a/s * b/s = (a * b) / (s * s) = (a * b / s) / s
  // Now s = 2^fracBits. This means that (a * b / s) = (a * b) >> fracBits!
  auto result = (io("a") * io("b")) >> fracBits;
  io("c") <<= regNext(result, "c");
}

void FixedHistogram::body() {
  auto index = regNext(io("index"));

  std::vector<FValue> entries;
  for (double prob : probabilities) {
    uint64_t bits = doubleToFixed(intBits, fracBits, prob);
    entries.push_back(cons(bits, uintType(intBits + fracBits)));
  }

  auto probVector = vector(entries);

  io("prob") <<= regNext(probVector[index]);
}

}

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
  using namespace mlir::spn::fpga::fixedpoint;

  initFirpContext(context.get(), "FixedHistogram");

  {
    //FixedAdd add(4, 4);
    //add.makeTop();

    std::vector<double> probabilities{
      0.0,
      0.25,
      0.5,
      0.75,
      1.0
    };

    FixedHistogram histogram(4, 4, 123, probabilities);
    histogram.makeTop();
  }

  firpContext()->finish();
  firpContext()->dump();

  return 0;
}