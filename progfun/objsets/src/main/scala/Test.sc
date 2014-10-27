def scalarProduct(xs: List[Double], ys: List[Double]) : Double = {for (i <- 0 until xs.size)yield xs(i) * ys(i)} sum

scalarProduct(List(1,2,3), List(3,2,1))