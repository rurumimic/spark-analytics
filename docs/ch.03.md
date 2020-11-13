# Chapter 3

## Knowledge

Spark docs: [Collaborative Filtering](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)

### Collaborative filtering: 협업 필터링

[Wikipedia](https://en.wikipedia.org/wiki/Collaborative_filtering)

### Factor analysis

[Wikipedia](https://en.wikipedia.org/wiki/Factor_analysis)

- Latent-factor: 잠재요인
- Observed Interaction: 상호작용
- Unobserved Underlying Reason: 숨은 원인
- Matrix Factorization: 행렬 분해
- Matrix Completion: 행렬 채우기
- Latent Feature: 잠재 특징

### Rank: 계수

[Wikipedia](https://en.wikipedia.org/wiki/Rank_(linear_algebra))

### Alternating Least Squares: 교차 최소 제곱

[Wikipeida](https://en.wikipedia.org/wiki/Matrix_completion#alternating_least_squares_minimization): Alternating least squares minimization

### Netflix Prize

[Wikipedia](https://en.wikipedia.org/wiki/Netflix_Prize)

- [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf): 암묵적 피드백 데이터셋에 대한 협업 필터링
- [Large-Scale Parallel Collaborative Filtering for the Netflix Prize](https://dl.acm.org/doi/10.1007/978-3-540-68880-8_32): 넷플릭스 프라이즈를 위한 대규모의 병렬 협업 필터링

### QR decomposition

[Wikipedia](https://en.wikipedia.org/wiki/QR_decomposition)

### Receiver Operating Characteristic: 수신자 조작 특성

[Wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

### Information retrieval

[Wikipedia](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))

- Precision
- Recall
- MeanAverage Precision

### Cross-validation

[Wikipedia](https://en.wikipedia.org/wiki/Cross-validation_(statistics))

- k-fold cross-validation

---

## Data

### Download

[https://storage.googleapis.com/aas-data-sets/profiledata_06-May-2005.tar.gz](https://storage.googleapis.com/aas-data-sets/profiledata_06-May-2005.tar.gz)

in `data/ch3`:

- artist_alias.txt: Artist name alias
- artist_data.txt: Artist ID 
- user_artist_data.txt: Users' play history

### Load

```scala
val rawUserArtistData = spark.read.textFile("data/ch3/user_artist_data.txt")
rawUserArtistData.take(5).foreach(println)
```

### ID min, max

`Int.MaxValue` = 2,147,483,647

```scala
val userArtistDF = rawUserArtistData.map { line =>
  val Array(user, artist, _*) = line.split(' ')
  (user.toInt, artist.toInt)
}.toDF("user", "artist")

userArtistDF.agg(min("user"), max("user"), min("artist"), max("artist")).show()

+---------+---------+-----------+-----------+
|min(user)|max(user)|min(artist)|max(artist)|
+---------+---------+-----------+-----------+
|       90|  2443548|          1|   10794401|
+---------+---------+-----------+-----------+
```

### Artist ID, Name

Load:

```scala
val rawArtistData = spark.read.textFile("data/ch3/artist_data.txt")
```

Raw data:

```scala
rawArtistData.take(5).foreach(println)

1134999 06Crazy Life
6821360 Pang Nakarin
10113088        Terfel, Bartoli- Mozart: Don
10151459        The Flaming Sidebur
6826647 Bodenstandig 3000
```

Map:

```scala
rawArtistData.map { line => 
  val (id, name) = line.span(_ != '\t')
  (id.toInt, name.trim)
}.count()

// Caused by: java.lang.NumberFormatException: For input string: "?"
```

FlatMap:

```scala
val artistByID = rawArtistData.flatMap { line =>
  val (id, name) = line.span(_ != '\t')
  if (name.isEmpty) {
    None
  } else {
    try {
      Some((id.toInt, name.trim))
    } catch {
      case _: NumberFormatException => None
    }
  }
}.toDF("id", "name")
```

```scala
artistByID.take(5).foreach(println)

[1134999,06Crazy Life]
[6821360,Pang Nakarin]
[10113088,Terfel, Bartoli- Mozart: Don]
[10151459,The Flaming Sidebur]
[6826647,Bodenstandig 3000]
```

### Alias

Load:

```scala
val rawArtistAlias = spark.read.textFile("data/ch3/artist_alias.txt")
```

Mapping:

```scala
val artistAlias = rawArtistAlias.flatMap { line =>
  val Array(artist, alias) = line.split('\t')
  if (artist.isEmpty) {
    None
  } else {
    Some((artist.toInt, alias.toInt))
  }
}.collect().toMap
```

```scala
artistAlias.head // (Int, Int) = (1208690,1003926)
artistByID.filter($"id" isin (1208690, 1003926)).show()

+-------+----------------+
|     id|            name|
+-------+----------------+
|1208690|Collective Souls|
|1003926| Collective Soul|
+-------+----------------+
```

---

## Modeling

### Helper

```scala
import org.apache.spark.sql._
import org.apache.spark.broadcast._

def buildCounts(
  rawUserArtistData: Dataset[String],
  bArtistAlias: Broadcast[Map[Int,Int]]
): DataFrame = {
  rawUserArtistData.map { line => 
    val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
    val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
    (userID, finalArtistID, count)
  }.toDF("user", "artist", "count")
}
```

### Broadcast Variable

Spark [doc](http://spark.apache.org/docs/latest/rdd-programming-guide.html#broadcast-variables)

```scala
val bArtistAlias = spark.sparkContext.broadcast(artistAlias)
val trainData = buildCounts(rawUserArtistData, bArtistAlias)
trainData.cache()
```

### ALS model

```scala
import org.apache.spark.ml.recommendation._
import scala.util.Random

val model = new ALS()
  .setSeed(Random.nextLong())
  .setImplicitPrefs(true)
  .setRank(10)
  .setRegParam(1.0)
  .setAlpha(1.0)
  .setMaxIter(5)
  .setUserCol("user")
  .setItemCol("artist")
  .setRatingCol("count")
  .setPredictionCol("prediction")
  .fit(trainData)

// model: org.apache.spark.ml.recommendation.ALSModel
```

### Feature Vector

```scala
model.userFactors.show(1, truncate = false)
```

| id | features |
|---|---|
| 90 | [-0.09313757, -0.08348628, -0.1380675, -0.23121716, -0.048566803, -0.06320441, 0.091850094, -0.08868533, -0.008541866, -0.05435346] |

### Test

```scala
val userID = 2093760
val existingArtistIDs = trainData
  .filter($"user" === userID)
  .select("artist")
  .as[Int]
  .collect() // Array(1180, 1255340, 378, 813, 942)
artistByID.filter($"id" isin (existingArtistIDs:_*)).show()

+-------+---------------+
|     id|           name|
+-------+---------------+
|   1180|     David Gray|
|    378|  Blackalicious|
|    813|     Jurassic 5|
|1255340|The Saw Doctors|
|    942|         Xzibit|
+-------+---------------+
```

### Recommend

- [recommendForAllUsers](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/recommendation/ALSModel.html#recommendForAllUsers-int-): Returns top numItems items recommended for each user, for all users.
- [lit](https://spark.apache.org/docs/latest/api/java/org/apache/spark/sql/functions.html#lit-java.lang.Object-): Creates a Column of literal value.

```scala
def makeRecommendations(model: ALSModel, userID: Int, howMany: Int): DataFrame = {
  val toRecommend = model
    .itemFactors
    .select($"id".as("artist"))
    .withColumn("user", lit(userID))
    
  model
    .transform(toRecommend)
    .select("artist", "prediction")
    .orderBy($"prediction".desc)
    .limit(howMany)
}
```

Enable implicit cartesian products by setting the configuration
variable:

```scala
spark.conf.set("spark.sql.crossJoin.enabled", "true")
```

```scala
val topRecommendations = makeRecommendations(model, userID, 5)
topRecommendations.show()

+-------+-----------+
| artist| prediction|
+-------+-----------+
|   2814|0.021783968|
|1001819|0.021308703|
|1300642|0.021292375|
|1037970|0.020558506|
|1007614|0.020497674|
+-------+-----------+
```

```scala
val recommendedArtistIDs = topRecommendations.select("artist").as[Int].collect()
artistByID.filter($"id" isin (recommendedArtistIDs:_*)).show()

+-------+----------+
|     id|      name|
+-------+----------+
|   2814|   50 Cent|
|1007614|     Jay-Z|
|1037970|Kanye West|
|1001819|      2Pac|
|1300642|  The Game|
+-------+----------+
```

#### recommendForUserSubset

```scala
import org.apache.spark.sql.types._
case class User(user: Int)
val userDS = Seq(User(userID)).toDS()
userDS.show()

val userRecommends = model.recommendForUserSubset(userDS, 5)
val topRecommends = userRecommends
  .select($"user", explode($"recommendations").alias("recommendation"))
  .select($"user", $"recommendation.*")

topRecommends.show()

+-------+-------+-----------+                                                   
|   user| artist|     rating|
+-------+-------+-----------+
|2093760|1300642|0.022198996|
|2093760|   2814|0.021402959|
|2093760|1001819|0.021072056|
|2093760|    829|0.020538548|
|2093760|1037970|0.020053174|
+-------+-------+-----------+
```

```scala
artistByID
  .filter($"id" isin (topRecommends.select("artist").as[Int].collect():_*))
  .show()

+-------+----------+
|     id|      name|
+-------+----------+
|   2814|   50 Cent|
|    829|       Nas|
|1037970|Kanye West|
|1001819|      2Pac|
|1300642|  The Game|
+-------+----------+
```

---

## AUC

def [areaUnderCurve](https://github.com/rurumimic/aas/blob/637d41f20626c870820395b7e791c04fd73c4116/ch03-recommender/src/main/scala/com/cloudera/datascience/recommender/RunRecommender.scala#L229)

```scala
import scala.collection.mutable.ArrayBuffer

def areaUnderCurve(
  positiveData: DataFrame,
  bAllArtistIDs: Broadcast[Array[Int]],
  predictFunction: (DataFrame => DataFrame)): Double = {

  // What this actually computes is AUC, per user. The result is actually something
  // that might be called "mean AUC".

  // Take held-out data as the "positive".
  // Make predictions for each of them, including a numeric score
  val positivePredictions = predictFunction(positiveData.select("user", "artist")).
    withColumnRenamed("prediction", "positivePrediction")

  // BinaryClassificationMetrics.areaUnderROC is not used here since there are really lots of
  // small AUC problems, and it would be inefficient, when a direct computation is available.

  // Create a set of "negative" products for each user. These are randomly chosen
  // from among all of the other artists, excluding those that are "positive" for the user.
  val negativeData = positiveData.select("user", "artist").as[(Int,Int)].
    groupByKey { case (user, _) => user }.
    flatMapGroups { case (userID, userIDAndPosArtistIDs) =>
      val random = new Random()
      val posItemIDSet = userIDAndPosArtistIDs.map { case (_, artist) => artist }.toSet
      val negative = new ArrayBuffer[Int]()
      val allArtistIDs = bAllArtistIDs.value
      var i = 0
      // Make at most one pass over all artists to avoid an infinite loop.
      // Also stop when number of negative equals positive set size
      while (i < allArtistIDs.length && negative.size < posItemIDSet.size) {
        val artistID = allArtistIDs(random.nextInt(allArtistIDs.length))
        // Only add new distinct IDs
        if (!posItemIDSet.contains(artistID)) {
          negative += artistID
        }
        i += 1
      }
      // Return the set with user ID added back
      negative.map(artistID => (userID, artistID))
    }.toDF("user", "artist")

  // Make predictions on the rest:
  val negativePredictions = predictFunction(negativeData).
    withColumnRenamed("prediction", "negativePrediction")

  // Join positive predictions to negative predictions by user, only.
  // This will result in a row for every possible pairing of positive and negative
  // predictions within each user.
  val joinedPredictions = positivePredictions.join(negativePredictions, "user").
    select("user", "positivePrediction", "negativePrediction").cache()

  // Count the number of pairs per user
  val allCounts = joinedPredictions.
    groupBy("user").agg(count(lit("1")).as("total")).
    select("user", "total")
  // Count the number of correctly ordered pairs per user
  val correctCounts = joinedPredictions.
    filter($"positivePrediction" > $"negativePrediction").
    groupBy("user").agg(count("user").as("correct")).
    select("user", "correct")

  // Combine these, compute their ratio, and average over all users
  val meanAUC = allCounts.join(correctCounts, Seq("user"), "left_outer").
    select($"user", (coalesce($"correct", lit(0)) / $"total").as("auc")).
    agg(mean("auc")).
    as[Double].first()

  joinedPredictions.unpersist()

  meanAUC
}
```

```scala
val allData = buildCounts(rawUserArtistData, bArtistAlias)
val Array(trainData, cvData) = allData.randomSplit(Array(0.9, 0.1))
trainData.cache()
cvData.cache()

val allArtistIDs = allData.select("artist").as[Int].distinct().collect()
val bAllArtistIDs = spark.sparkContext.broadcast(allArtistIDs)

val model = new ALS()
  .setSeed(Random.nextLong())
  .setImplicitPrefs(true)
  .setRank(10)
  .setRegParam(0.01)
  .setAlpha(1.0)
  .setMaxIter(5)
  .setUserCol("user")
  .setItemCol("artist")
  .setRatingCol("count")
  .setPredictionCol("prediction")
  .fit(trainData)
```

```scala
areaUnderCurve(cvData, bAllArtistIDs, model.transform)

Double = 0.9018833580535511
```

AUC: 0.9018833580535511

```scala
def predictMostListened(train: DataFrame)(allData: DataFrame): DataFrame = {
  val listenCounts = train.groupBy("artist").
    agg(sum("count").as("prediction")).
    select("artist", "prediction")
  allData.
    join(listenCounts, Seq("artist"), "left_outer").
    select("user", "artist", "prediction")
}
```

```scala
areaUnderCurve(cvData, bAllArtistIDs, predictMostListened(trainData))

Double = 0.8764452188648839
```

AUC: 0.8764452188648839

---

## Hyperparameter

```scala
val evaluations =
  for (rank     <- Seq(5,  30);
        regParam <- Seq(1.0, 0.0001);
        alpha    <- Seq(1.0, 40.0))
  yield {
    val model = new ALS().
      setSeed(Random.nextLong()).
      setImplicitPrefs(true).
      setRank(rank).setRegParam(regParam).
      setAlpha(alpha).setMaxIter(20).
      setUserCol("user").setItemCol("artist").
      setRatingCol("count").setPredictionCol("prediction").
      fit(trainData)

    val auc = areaUnderCurve(cvData, bAllArtistIDs, model.transform)

    model.userFactors.unpersist()
    model.itemFactors.unpersist()

    (auc, (rank, regParam, alpha))
  }
```

```scala
evaluations.sorted.reverse.foreach(println)

(0.9130566657276805,(30,1.0,40.0))
(0.9123604434928522,(30,1.0E-4,40.0))
(0.9103777923152742,(5,1.0,40.0))
(0.9094360808356605,(5,1.0E-4,40.0))
(0.9057413968217879,(5,1.0,1.0))
(0.9047730358471702,(5,1.0E-4,1.0))
(0.9042031255288289,(30,1.0,1.0))
(0.8953671889893808,(30,1.0E-4,1.0))
```

---

## Best model's recommendations

```scala
val model = new ALS()
  .setSeed(Random.nextLong())
  .setImplicitPrefs(true)
  .setRank(30)
  .setRegParam(1.0)
  .setAlpha(40.0)
  .setMaxIter(20)
  .setUserCol("user")
  .setItemCol("artist")
  .setRatingCol("count")
  .setPredictionCol("prediction")
  .fit(trainData)

val topRecommendations = makeRecommendations(model, userID, 5)
val recommendedArtistIDs = topRecommendations.select("artist").as[Int].collect()
artistByID.filter($"id" isin (recommendedArtistIDs:_*)).show()
```

```scala
+-------+-----------+
|     id|       name|
+-------+-----------+
|1034635|  [unknown]|
|1000113|The Beatles|
|    930|     Eminem|
|    976|    Nirvana|
|   4267|  Green Day|
+-------+-----------+
```
