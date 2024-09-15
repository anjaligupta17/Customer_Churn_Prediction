from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType

# Step 1: Create Spark session
spark = SparkSession.builder \
    .appName("Customer Churn EDA") \
    .getOrCreate()

# Step 2: Load the CSV file
file_path = "/Users/anjaligupta/Desktop/project/project_1"
data = spark.read.csv(file_path, header=True, inferSchema=True)

# Step 3: Display the schema and initial data
data.printSchema()
data.show(5)

# Step 4: Data Cleaning and Transformation
# Convert categorical columns to numerical
data = data.withColumn("Int'l Plan", F.when(F.col("Int'l Plan") == 'yes', 1).otherwise(0)) \
           .withColumn("VMail Plan", F.when(F.col("VMail Plan") == 'yes', 1).otherwise(0)) \
           .withColumn("Churn", F.when(F.col("Churn") == 'True.', 1).otherwise(0))

# Ensure correct data types for columns
data = data.withColumn("Account Length", F.col("Account Length").cast(IntegerType())) \
           .withColumn("Day Mins", F.col("Day Mins").cast(DoubleType())) \
           .withColumn("Day Charge", F.col("Day Charge").cast(DoubleType())) \
           .withColumn("Eve Mins", F.col("Eve Mins").cast(DoubleType())) \
           .withColumn("Eve Charge", F.col("Eve Charge").cast(DoubleType())) \
           .withColumn("Night Mins", F.col("Night Mins").cast(DoubleType())) \
           .withColumn("Night Charge", F.col("Night Charge").cast(DoubleType())) \
           .withColumn("Intl Mins", F.col("Intl Mins").cast(DoubleType())) \
           .withColumn("Intl Charge", F.col("Intl Charge").cast(DoubleType())) \
           .withColumn("CustServ Calls", F.col("CustServ Calls").cast(IntegerType()))

# Step 5: Feature Engineering - create total minutes and total charges
data = data.withColumn("Total Mins", F.col("Day Mins") + F.col("Eve Mins") + F.col("Night Mins") + F.col("Intl Mins")) \
           .withColumn("Total Charge", F.col("Day Charge") + F.col("Eve Charge") + F.col("Night Charge") + F.col("Intl Charge"))

# Step 6: Summary Statistics
summary_stats = data.describe()
summary_stats.write.csv("/Users/anjaligupta/Desktop/project/eda_summary_statistics.csv", header=True)

# Step 7: Data Distribution
# Count of each churn class
data_distribution = data.groupBy("Churn").count()
data_distribution.write.csv("/Users/anjaligupta/Desktop/project/eda_data_distribution.csv", header=True)

# Summary of numeric columns
numeric_columns = ["Account Length", "Day Mins", "Day Charge", "Eve Mins", "Eve Charge", 
                   "Night Mins", "Night Charge", "Intl Mins", "Intl Charge", 
                   "CustServ Calls", "Total Mins", "Total Charge"]

# Collect summary of numeric columns
summary_numeric_stats = []
for col in numeric_columns:
    summary = data.select(F.avg(col).alias(f"Average {col}"),
                          F.min(col).alias(f"Min {col}"),
                          F.max(col).alias(f"Max {col}"),
                          F.stddev(col).alias(f"StdDev {col}")).toPandas()
    summary_numeric_stats.append(summary)

# Convert to a single DataFrame
import pandas as pd
summary_numeric_df = pd.concat(summary_numeric_stats, axis=1)
summary_numeric_spark_df = spark.createDataFrame(summary_numeric_df)
summary_numeric_spark_df.write.csv("/Users/anjaligupta/Desktop/project/eda_numeric_statistics.csv", header=True)

# Step 8: Correlation Analysis
correlations = [(col, data.stat.corr(col, "Churn")) for col in numeric_columns]
correlation_df = spark.createDataFrame(correlations, ["Feature", "Correlation with Churn"])
correlation_df.write.csv("/Users/anjaligupta/Desktop/project/eda_correlation.csv", header=True)

# Stop the Spark session
spark.stop()
