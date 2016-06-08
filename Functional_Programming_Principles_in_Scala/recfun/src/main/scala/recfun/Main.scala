package recfun

object Main {
  def main(args: Array[String]) {
    println("Pascal's Triangle")
    for (row <- 0 to 10) {
      for (col <- 0 to row)
        print(pascal(col, row) + " ")
      println()
    }
  }

  /**
   * Exercise 1
   */
    def pascal(c: Int, r: Int): Int = {
      if (c == 0 || c == r) 
        1
      else 
        pascal(c - 1, r - 1) + pascal(c, r - 1)
    }
  
  /**
   * Exercise 2
   */
    def balance(chars: List[Char]): Boolean = {
      
  	  def update(chars: List[Char], num: Int): Boolean = {
  	    var tempNum = 0
  	    
  	    if (chars.isEmpty)
  	      if (num == 0) true
  	      else false
  	    else {
  	      if (chars.head == '(') tempNum = num + 1
  	      else if (chars.head == ')') tempNum = num - 1
  	      else tempNum = num
  	      
  	      if (tempNum >= 0) update(chars.tail, tempNum)
  	      else false
  	      }
  	  }
  	  update(chars, 0)
    }
  
  /**
   * Exercise 3
   */
    def countChange(money: Int, coins: List[Int]): Int = {
      if (coins.isEmpty)
        0
      else if (money < coins.head)
        countChange(money, coins.tail)
      else if (money == coins.head)
        1 + countChange(money, coins.tail)
      else
        countChange(money - coins.head, coins) + countChange(money, coins.tail)
    }

  }
