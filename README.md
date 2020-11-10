# Spark Analytics

- Book: [Advanced Analytics with Spark](https://github.com/sryza/aas)
- [Apache Spark](https://spark.apache.org/)
- Docs: [Latest](https://spark.apache.org/docs/latest/)
  - [2.4.7](https://spark.apache.org/docs/2.4.7/)

---

## Usage

### Download

Index of [/apache/spark](http://mirror.apache-kr.org/apache/spark)

```bash
curl -O http://mirror.apache-kr.org/apache/spark/spark-2.4.7/spark-2.4.7-bin-hadoop2.7.tgz
```

### Environment variables

```bash
export SPARK_HOME=$(pwd)/download/spark-2.4.7-bin-hadoop2.7
export PATH=$PATH:$SPARK_HOME/bin
export PATH=$PATH:$SPARK_HOME/sbin
```

### Java version

`java -version`

```bash
java version "1.8.0_221"
Java(TM) SE Runtime Environment (build 1.8.0_221-b11)
Java HotSpot(TM) 64-Bit Server VM (build 25.221-b11, mixed mode)
```

### Spark Shell

```bash
spark-shell --master "local[*]" --driver-memory 4g
```

```bash
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _  / __/   _/
   /___/ .__/\_,_/_/ /_/\_\   version 2.4.7
      /_/

Using Scala version 2.11.12 (Java HotSpot(TM) 64-Bit Server VM, Java 1.8.0_221)

scala> :quit
```

---
