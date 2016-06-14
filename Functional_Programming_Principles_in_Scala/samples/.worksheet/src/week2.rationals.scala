package week2

object rationals {;import org.scalaide.worksheet.runtime.library.WorksheetSupport._; def main(args: Array[String])=$execute{;$skip(63); 

  val x = new Rational(1, 3);System.out.println("""x  : week2.Rational = """ + $show(x ));$skip(29); 
  val y = new Rational(5, 7);System.out.println("""y  : week2.Rational = """ + $show(y ));$skip(29); 
  val z = new Rational(3, 2);System.out.println("""z  : week2.Rational = """ + $show(z ));$skip(13); val res$0 = 
  
  x.numer;System.out.println("""res0: Int = """ + $show(res$0));$skip(10); val res$1 = 
  x.denom;System.out.println("""res1: Int = """ + $show(res$1));$skip(8); val res$2 = 
  y + y;System.out.println("""res2: week2.Rational = """ + $show(res$2));$skip(12); val res$3 = 
  x - y - z;System.out.println("""res3: week2.Rational = """ + $show(res$3));$skip(8); val res$4 = 
  x < y;System.out.println("""res4: Boolean = """ + $show(res$4));$skip(10); val res$5 = 
  x max y;System.out.println("""res5: week2.Rational = """ + $show(res$5));$skip(21); val res$6 = 
  
  new Rational(2);System.out.println("""res6: week2.Rational = """ + $show(res$6));$skip(21); val res$7 = 
  new Rational(4, 5);System.out.println("""res7: week2.Rational = """ + $show(res$7));$skip(36); val res$8 = 
  
  new Rational(1045540, 5232500);System.out.println("""res8: week2.Rational = """ + $show(res$8))}
  
}

class Rational(x: Int, y: Int) {

  require(y != 0, "denominator must be nonzero")
  
  def this(x: Int) = this(x, 1)
  
  private def gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)
  private val g = gcd(x, y)
  
  def numer = x
  def denom = y
  
  def +(that: Rational) =
    new Rational(
      numer * that.denom + that.numer * denom, denom * that.denom)
      
  def unary_- : Rational = new Rational(-this.numer, this.denom)
  
  def -(that: Rational) = this + -that
  
  def <(that: Rational) = this.numer * that.denom < that.numer * this.denom
  
  def max(that: Rational) = if (this < that) that else this
  
      
  override def toString = numer / g + "/" + denom / g
  
}
