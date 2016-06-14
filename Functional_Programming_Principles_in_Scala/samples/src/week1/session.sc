package week1

object session {

  def abs(x: Double) = if (x < 0) -x else x       //> abs: (x: Double)Double
  
  def sqrt(x: Double) = {
  
	  def sqrtIter(guess: Double): Double =
	    if (isGoodEnough(guess)) guess
	    else sqrtIter(improve(guess))
	    
	  def isGoodEnough(guess: Double) =
	    abs(guess * guess - x) < 0.001 * x
	    
	  def improve(guess: Double) =
	    (guess + x / guess) / 2
	    
    sqrtIter(1.0)
  }                                               //> sqrt: (x: Double)Double
  
  
  sqrt(0.1e-20)                                   //> res0: Double = 3.1633394544890125E-11
  sqrt(4)                                         //> res1: Double = 2.000609756097561
  sqrt(1e-6)                                      //> res2: Double = 0.0010000001533016628
  sqrt(1e60)                                      //> res3: Double = 1.0000788456669446E30
  
  3 % 1                                           //> res4: Int(0) = 0
  
}