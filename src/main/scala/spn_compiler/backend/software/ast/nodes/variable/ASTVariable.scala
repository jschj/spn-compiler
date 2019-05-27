package spn_compiler.backend.software.ast.nodes.variable

import spn_compiler.backend.software.ast.nodes.reference.ASTReferencable
import spn_compiler.backend.software.ast.nodes.statement.variable.ASTVariableDeclaration
import spn_compiler.backend.software.ast.nodes.types.ASTType

class ASTVariable private[ast] (val ty : ASTType, val name : String) extends ASTReferencable {

  private var _declaration : Option[ASTVariableDeclaration] = None

  def isDeclared : Boolean = _declaration.isDefined

  def declaration : ASTVariableDeclaration = _declaration.orNull

  def declaration_=(declare : ASTVariableDeclaration) : Unit = _declaration = Some(declare)

  def getType : ASTType = ty

}

object ASTVariable {

  def unapply(arg : ASTVariable)
    : Option[(ASTType, String)] = Some((arg.ty, arg.name))

}