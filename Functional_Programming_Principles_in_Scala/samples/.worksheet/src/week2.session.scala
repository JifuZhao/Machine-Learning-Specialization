package week2

object session {;import org.scalaide.worksheet.runtime.library.WorksheetSupport._; def main(args: Array[String])=$execute{;$skip(411); 

/**
  def sum(f: Int => Int, a: Int, b: Int) = {
    def loop(a: Int, acc: Int): Int =
      if (a > b) acc
      else loop(a+1, f(a) + acc)
      
    loop(a, 0)
  }
  
  sum((x: Int) => x*x, 3, 5)
*/
  def mapReduce(f: Int => Int, combine: (Int, Int) => Int, zero: Int)(a: Int, b: Int): Int =
      if (a > b) zero
      else combine(f(a), mapReduce(f, combine, zero)(a+1, b));System.out.println("""mapReduce: (f: Int => Int, combine: (Int, Int) => Int, zero: Int)(a: Int, b: Int)Int""");$skip(96); 
      
  def product(f: Int => Int)(a: Int, b: Int): Int = mapReduce(f, (x, y) => x*y, 1)(a, b);System.out.println("""product: (f: Int => Int)(a: Int, b: Int)Int""");$skip(31); val res$0 = 
		 
	product(x => x * x)(3, 4);System.out.println("""res0: Int = """ + $show(res$0));$skip(85); 
		 
  def g(f: (Int, Int, Int) => Int)(a: Int, b: Int, c: Int): Int =
    f(a, b, c);System.out.println("""g: (f: (Int, Int, Int) => Int)(a: Int, b: Int, c: Int)Int""");$skip(54); 
    
  def f(a: Int, b: Int, c: Int): Int = a + b + c;System.out.println("""f: (a: Int, b: Int, c: Int)Int""");$skip(36); val res$1 = 
  
  g((x, y, z) => x*y*z)(1, 2, 4);System.out.println("""res1: Int = """ + $show(res$1));$skip(18); val res$2 = 
 
  g(f)(1, 2, 4);System.out.println("""res2: Int = """ + $show(res$2));$skip(47); 
  
  def s = (x: Int, y: Int, z: Int) => x*y*z;System.out.println("""s: => (Int, Int, Int) => Int""");$skip(13); val res$3 = 
  s(1, 2, 3);System.out.println("""res3: Int = """ + $show(res$3))}
  
}
