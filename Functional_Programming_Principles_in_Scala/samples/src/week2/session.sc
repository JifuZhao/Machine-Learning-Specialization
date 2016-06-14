package week2

object session {

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
      else combine(f(a), mapReduce(f, combine, zero)(a+1, b))
                                                  //> mapReduce: (f: Int => Int, combine: (Int, Int) => Int, zero: Int)(a: Int, b:
                                                  //|  Int)Int
      
  def product(f: Int => Int)(a: Int, b: Int): Int = mapReduce(f, (x, y) => x*y, 1)(a, b)
                                                  //> product: (f: Int => Int)(a: Int, b: Int)Int
		 
	product(x => x * x)(3, 4)                 //> res0: Int = 144
		 
  def g(f: (Int, Int, Int) => Int)(a: Int, b: Int, c: Int): Int =
    f(a, b, c)                                    //> g: (f: (Int, Int, Int) => Int)(a: Int, b: Int, c: Int)Int
    
  def f(a: Int, b: Int, c: Int): Int = a + b + c  //> f: (a: Int, b: Int, c: Int)Int
  
  g((x, y, z) => x*y*z)(1, 2, 4)                  //> res1: Int = 8
 
  g(f)(1, 2, 4)                                   //> res2: Int = 7
  
  def s = (x: Int, y: Int, z: Int) => x*y*z       //> s: => (Int, Int, Int) => Int
  s(1, 2, 3)                                      //> res3: Int = 6
  
}