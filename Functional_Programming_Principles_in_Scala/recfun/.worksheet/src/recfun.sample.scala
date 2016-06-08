package recfun

object sample {;import org.scalaide.worksheet.runtime.library.WorksheetSupport._; def main(args: Array[String])=$execute{;$skip(60); 
  
  var chars = "()".toList;System.out.println("""chars  : List[Char] = """ + $show(chars ));$skip(460); 

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

	};System.out.println("""balance: (chars: List[Char])Boolean""");$skip(20); val res$0 = 
  
  balance(chars);System.out.println("""res0: Boolean = """ + $show(res$0))}
}
