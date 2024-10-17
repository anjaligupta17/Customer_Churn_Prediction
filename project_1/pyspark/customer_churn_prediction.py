from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import pandas as pd

# Step 1: Create Spark session
spark = SparkSession.builder \
    .appName("Customer Churn EDA") \
    .getOrCreate()

# Step 2: Load the CSV file with specified schema
file_path = "/Users/anjaligupta/Desktop/project/project_1"
schema = StructType([
  StructField("Customer ID", StringType(), True),
  StructField("State", StringType(), True),
  StructField("Account Length", IntegerType(), True),
  StructField("Area Code", StringType(), True),
  StructField("Phone", StringType(), True),
  StructField("Int'l Plan", StringType(), True),  # You can rename this later to avoid special characters
  StructField("VMail Plan", StringType(), True),
  StructField("VMail Message", IntegerType(), True),
  StructField("Day Mins", DoubleType(), True),
  StructField("Day Calls", IntegerType(), True),
  StructField("Day Charge", DoubleType(), True),
  StructField("Eve Mins", DoubleType(), True),
  StructField("Eve Calls", IntegerType(), True),
  StructField("Eve Charge", DoubleType(), True),
  StructField("Night Mins", DoubleType(), True),
  StructField("Night Calls", IntegerType(), True),
  StructField("Night Charge", DoubleType(), True),
  StructField("Intl Mins", DoubleType(), True),
  StructField("Intl Calls", IntegerType(), True),
  StructField("Intl Charge", DoubleType(), True),
  StructField("CustServ Calls", IntegerType(), True),
  StructField("Churn", StringType(), True)
])

data = spark.read.csv(file_path, header=True, schema=schema)

# Step 3: Display the schema and initial data
data.printSchema()
data.show(5)

# Step 4: Data Cleaning and Transformation
# Convert categorical columns to numerical
# Rename the Int'l Plan to avoid special character issues
data = data.withColumnRenamed("Int'l Plan", "Intl Plan")

data = data.withColumn("Intl Plan", F.when(F.col("Intl Plan") == 'yes', 1).otherwise(0)) \
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
summary_stats.write.csv("/Users/anjaligupta/Desktop/project/pyspark/eda_summary_statistics.csv", header=True)

# Step 7: Data Distribution
# Count of each churn class
data_distribution = data.groupBy("Churn").count()
data_distribution.write.csv("/Users/anjaligupta/Desktop/project/pyspark/eda_data_distribution.csv", header=True)

# Step 8: Summary of numeric columns
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

# Convert to a single DataFrame using Pandas
summary_numeric_df = pd.concat(summary_numeric_stats, axis=1)

# Save as CSV
summary_numeric_df.to_csv("/Users/anjaligupta/Desktop/project/pyspark/eda_numeric_statistics.csv", index=False)

# Step 9: Correlation Analysis
correlations = [(col, data.stat.corr(col, "Churn")) for col in numeric_columns]
correlation_df = spark.createDataFrame(correlations, ["Feature", "Correlation with Churn"])
correlation_df.write.csv("/Users/anjaligupta/Desktop/project/pyspark/eda_correlation.csv", header=True)

# Stop the Spark session
spark.stop()
