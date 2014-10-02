package funsets

import org.scalatest.FunSuite

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

/**
 * This class is a test suite for the methods in object FunSets. To run
 * the test suite, you can either:
 *  - run the "test" command in the SBT console
 *  - right-click the file in eclipse and chose "Run As" - "JUnit Test"
 */
@RunWith(classOf[JUnitRunner])
class FunSetSuite extends FunSuite {


  /**
   * Link to the scaladoc - very clear and detailed tutorial of FunSuite
   *
   * http://doc.scalatest.org/1.9.1/index.html#org.scalatest.FunSuite
   *
   * Operators
   *  - test
   *  - ignore
   *  - pending
   */

  /**
   * Tests are written using the "test" operator and the "assert" method.
   */
  test("string take") {
    val message = "hello, world"
    assert(message.take(5) == "hello")
  }

  /**
   * For ScalaTest tests, there exists a special equality operator "===" that
   * can be used inside "assert". If the assertion fails, the two values will
   * be printed in the error message. Otherwise, when using "==", the test
   * error message will only say "assertion failed", without showing the values.
   *
   * Try it out! Change the values so that the assertion fails, and look at the
   * error message.
   */
  test("adding ints") {
    assert(1 + 2 === 3)
  }

  
  import FunSets._

  test("contains is implemented") {
    assert(contains(x => true, 100))
  }
  
  /**
   * When writing tests, one would often like to re-use certain values for multiple
   * tests. For instance, we would like to create an Int-set and have multiple test
   * about it.
   * 
   * Instead of copy-pasting the code for creating the set into every test, we can
   * store it in the test class using a val:
   * 
   *   val s1 = singletonSet(1)
   * 
   * However, what happens if the method "singletonSet" has a bug and crashes? Then
   * the test methods are not even executed, because creating an instance of the
   * test class fails!
   * 
   * Therefore, we put the shared values into a separate trait (traits are like
   * abstract classes), and create an instance inside each test method.
   * 
   */

  trait TestSets {
    val s1 = singletonSet(1)
    val s2 = singletonSet(2)
    val s3 = singletonSet(3)

    val s123 = union(union(s1,s2),s3)
  }

  /**
   * This test is currently disabled (by using "ignore") because the method
   * "singletonSet" is not yet implemented and the test would fail.
   * 
   * Once you finish your implementation of "singletonSet", exchange the
   * function "ignore" by "test".
   */
  test("singletonSet(1) contains 1") {
    
    /**
     * We create a new instance of the "TestSets" trait, this gives us access
     * to the values "s1" to "s3". 
     */
    new TestSets {
      /**
       * The string argument of "assert" is a message that is printed in case
       * the test fails. This helps identifying which assertion failed.
       */
      assert(contains(s1, 1), "Singleton")
    }
  }

  test("union contains all elements") {
    new TestSets {
      val s = union(s1, s2)
      assert(contains(s, 1), "Union 1")
      assert(contains(s, 2), "Union 2")
      assert(!contains(s, 3), "Union 3")
    }
  }

  test("intersect"){
    new TestSets {
      val s = intersect(s1, s2)
      val x = intersect(s123, s2)
      val y = intersect(s123, singletonSet(5))
      assert(!contains(s,1), "Intersect1")
      assert(!contains(s,2), "Intersect2")
      assert(contains(x,2), "Intersect3")
      assert(!contains(x,1), "Intersect4")
      assert(!contains(y,1), "Intersect5")
    }
  }
  test("diff"){
    new TestSets {
      val p = union(s1, s2)
      val q = union(s2,s3)

      val d = diff(p,q)

      assert(contains(d,1),"Diff1")
      assert(!contains(d,2),"Diff2")
      assert(!contains(d,3),"Diff3")
    }
  }

  test("filter"){
    new TestSets {
      val f = filter(s123, (x => (x%3 == 0)))
      assert(contains(f,3), "Filter1")
      assert(!contains(f,1), "Filter2")
      assert(!contains(f,2), "Filter3")
    }
  }

  test("forall"){
    new TestSets {
      val f = union(union(union(s123, singletonSet(4)), singletonSet(5)), singletonSet(6))
      val tr = forall(f, x => (x > 0))
      val fa = forall(f, x => (x > 1))
      assert(tr, "Forall1")
      assert(!fa,"Forall2")
    }
  }

  test("exists"){
    new TestSets {
      val f = union(union(union(s123, singletonSet(4)), singletonSet(5)), singletonSet(6))
      val tr = exists(f, (x => (x==3)))
      val fa = exists(f, (x => (x == 0)))
      val tr1 = exists(f, (x => (x > 0)))

      assert(tr, "Exists1")
      assert(tr1, "Exists2")
      assert(!fa, "Exists3")
    }
  }

  test("map"){
    new TestSets {
      val f = union(union(union(s123, singletonSet(4)), singletonSet(5)), singletonSet(6))
      val n = map(f, (x => (x % 2)))
      assert(contains(n,1), "Map1")
      assert(contains(n,0), "Map2")
      assert(!contains(n,2), "Map3")
    }
  }
}
