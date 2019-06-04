package spn_compiler.driver.config

import java.io.File

trait CompilerConfig[R <: CLIConfig[R]] extends CLIConfig[R] {

  private var optLevel : Int = 0
  def setOptimizationLevel(level : Int) : R = {
    optLevel = Math.max(optLevel, level)
    self
  }
  def optimizationLevel : Int = optLevel

  private var outFile : File = new File("spn.out")
  def setOutFile(file : File) : R = {
    outFile = file
    self
  }
  def outputFile : File = outFile

  private var codeOnly : Boolean = false
  def setCodeOnly(bool : Boolean) : R = {
    codeOnly = bool
    self
  }
  def outputCodeOnly : Boolean = codeOnly

  private var fastMath : Boolean = false
  def enableFastMath(bool : Boolean) : R = {
    fastMath = bool
    self
  }
  def isFastMathEnabled : Boolean = fastMath

}
