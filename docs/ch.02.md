# Chapter 02

## Basic

```scala
val rawblocks = sc.textFile("data/linkage")
```

### first, collect, take

```scala
rawblocks.first // String = "id_1","id_2","cmp_fname_c1" ...

val head = rawblocks.take(10) // Array[String] = Array(...)
head.length // 10
```

### foreach, println

```scala
head.foreach(println)
```

### [Higher-order Function](https://en.wikipedia.org/wiki/Higher-order_function)

a function that does at least one of the following:

- takes one or more functions as arguments (i.e. procedural parameters),
- returns a function as its result.

### filter

```scala
def isHeader(line: String) = line.contains("id_1")
// or
def isHeader(line: String): Boolean = {
  line.contains("id_1")
}
```

```scala
head.filter(isHeader).foreach(println)
head.filterNot(isHeader).foreach(println)
head.filter(x => !isHeader(x)).length
head.filter(!isHeader(_)).length
```

### Remove Header

```bash
cat linkage/block_* | sed '1d' > linkage.csv
```

---

## DataFrame

[Spark SQL, DataFrames and Datasets Guide](https://spark.apache.org/docs/2.4.7/sql-programming-guide.html)

- A Dataset is a distributed collection of data. added in Spark 1.6.
- A DataFrame is a Dataset organized into named columns. 

### spark

- SparkSession was SQLContext in Spark 1.3.
- SparkSession is a wrapper of SparkContext.

```scala
spark // org.apache.spark.sql.SparkSession

spark.sparkContext // org.apache.spark.SparkContext
sc // org.apache.spark.SparkContext
```

### Reader API

```scala
val prev = spark.read.csv("data/linkage") // org.apache.spark.sql.DataFrame = [_c0: string, _c1: string ...
prev.show()
```

`:paste` + `ctrl-D`:

```scala
val parsed = spark.read
  .option("header", "true")
  .option("nullValue", "?")
  .option("inferSchema", "true")
  .csv("data/linkage")
```

```scala
parsed.printSchema()

/*
root
 |-- id_1: integer (nullable = true)
 |-- id_2: integer (nullable = true)
 |-- cmp_fname_c1: double (nullable = true)
 |-- cmp_fname_c2: double (nullable = true)
 |-- cmp_lname_c1: double (nullable = true)
 |-- cmp_lname_c2: double (nullable = true)
 |-- cmp_sex: integer (nullable = true)
 |-- cmp_bd: integer (nullable = true)
 |-- cmp_bm: integer (nullable = true)
 |-- cmp_by: integer (nullable = true)
 |-- cmp_plz: integer (nullable = true)
 |-- is_match: boolean (nullable = true)
*/
```

#### Faster

```scala
import org.apache.spark.sql.types._

val schema = StructType(Array(
  StructField("id_1", IntegerType, true),
  StructField("id_2", IntegerType, true),
  StructField("cmp_fname_c1", DoubleType, true),
  StructField("cmp_fname_c2", DoubleType, true),
  StructField("cmp_lname_c1", DoubleType, true),
  StructField("cmp_lname_c2", DoubleType, true),
  StructField("cmp_sex", IntegerType, true),
  StructField("cmp_bd", IntegerType, true),
  StructField("cmp_bm", IntegerType, true),
  StructField("cmp_by", IntegerType, true),
  StructField("cmp_plz", IntegerType, true),
  StructField("is_match", BooleanType, true)
))

val parsed = spark.read
  .option("header", "true")
  .option("nullValue", "?")
  .schema(schema)
  .csv("data/linkage")

parsed.printSchema()
parsed.show()
```

### Analytics

```scala
parsed.count() // 5749132
parsed.cache()
```

#### RDD

```scala
parsed.rdd
  .map(_.getAs[Boolean]("is_match"))
  .countByValue()

// scala.collection.Map[Boolean,Long]
// = Map(true -> 20931, false -> 5728201)
```

#### DataFrame

```scala
parsed
  .groupBy("is_match")
  .count()
  .orderBy($"count".desc)
  .show()

/*
+--------+-------+
|is_match|  count|
+--------+-------+
|   false|5728201|
|    true|  20931|
+--------+-------+
*/
```

```scala
parsed.agg(avg($"cmp_sex"), stddev($"cmp_sex")).show()

/*
+-----------------+--------------------+
|     avg(cmp_sex)|stddev_samp(cmp_sex)|
+-----------------+--------------------+
|0.955001381078048| 0.20730111116897834|
+-----------------+--------------------+
*/
```

#### Spark SQL

Create `linkage` view:

```scala
parsed.createOrReplaceTempView("linkage")
```

Query:

```scala
spark.sql("""
  SELECT is_match, COUNT(*) cnt
  FROM linkage
  GROUP BY is_match
  ORDER BY cnt DESC
""").show()
```

#### describe

count, mean, stddev, min, max:

```scala
val summary = parsed.describe()
summary.show()
summary.select("summary", "cmp_fname_c1", "cmp_fname_c2").show()
```

`where` alias `filter`:

```scala
val matches = parsed.where("is_match = true") // Spark SQL
val matchSummary = matches.describe()

val misses = parsed.filter($"is_match" === false) // DataFrame API
val missSummary = misses.describe()
```

#### Pivoting, Reshaping

```scala
val schema = summary.schema
val longForm = summary.flatMap(row => {
  val metric = row.getString(0)
  (1 until row.size).map(i => {
    (metric, schema(i).name, row.getString(i).toDouble)
  })
})
```

```scala
val longDF = longForm.toDF("metric", "field", "value")
longDF.show()
```

```scala
val wideDF = longDF
  .groupBy("field")
  .pivot("metric", Seq("count", "mean", "stddev", "min", "max"))
  .agg(first("value"))
wideDF.select("field", "count", "mean").show()
```

#### match and miss summaries

```scala
:load src/ch02/Pivot.scala
```

```scala
val matchSummaryT = pivotSummary(matchSummary)
val missSummaryT = pivotSummary(missSummary)
```

#### Join by SQL

```scala
matchSummaryT.createOrReplaceTempView("match_desc")
missSummaryT.createOrReplaceTempView("miss_desc")

spark.sql("""
  SELECT a.field, a.count + b.count total, a.mean - b.mean delta
  FROM match_desc a INNER JOIN miss_desc b ON a.field = b.field
  WHERE a.field NOT IN ("id_1", "id_2")
  ORDER BY delta DESC, total DESC
""").show()
```

#### Case Class

```scala
case class MatchData(
  id_1: Int,
  id_2: Int,
  cmp_fname_c1: Option[Double],
  cmp_fname_c2: Option[Double],
  cmp_lname_c1: Option[Double],
  cmp_lname_c2: Option[Double],
  cmp_sex: Option[Int],
  cmp_bd: Option[Int],
  cmp_bm: Option[Int],
  cmp_by: Option[Int],
  cmp_plz: Option[Int],
  is_match: Boolean
)
```

#### to DataSet

```scala
val matchData = parsed.as[MatchData]
matchData.show()
```

#### Helper Case Class

```scala
case class Score(value: Double) {
  def +(oi: Option[Int]) = {
    Score(value + oi.getOrElse(0))
  }
}
```

```scala
def scoreMatchData(md: MatchData): Double = {
  (Score(md.cmp_lname_c1.getOrElse(0.0)) + md.cmp_plz
    + md.cmp_by + md.cmp_bd + md.cmp_bm).value
}
```

```scala
val scored = matchData.map { md =>
  (scoreMatchData(md), md.is_match)
}.toDF("score", "is_match")
```

#### Model

```scala
def crossTabs(scored: DataFrame, t: Double): DataFrame = {
  scored.selectExpr(s"score >= $t as above", "is_match")
  .groupBy("above")
  .pivot("is_match", Seq("true", "false"))
  .count()
}
```

```scala
crossTabs(scored, 4.0).show()

/*
+-----+-----+-------+
|above| true|  false|
+-----+-----+-------+
| true|20871|    637|
|false|   60|5727564|
+-----+-----+-------+
*/
```

```scala
crossTabs(scored, 2.0).show()

/*
+-----+-----+-------+
|above| true|  false|
+-----+-----+-------+
| true|20931| 596414|
|false| null|5131787|
+-----+-----+-------+
*/
```
