from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, mean

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Customer Churn Analysis") \
    .getOrCreate()

# Load data into a DataFrame
csv_file_path = "/Users/anjaligupta/Desktop/project/project_1/pyspark/churn.csv"
df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

# Show the first 5 rows
df.show(5)

# Print schema to check data types
df.printSchema()

# Total number of customers
total_customers = df.count()
print(f"Total customers: {total_customers}")

# Count number of churned customers
churned_customers = df.filter(col("Churn") == "True.").count()
print(f"Number of churned customers: {churned_customers}")

# Churn rate
churn_rate = churned_customers / total_customers
print(f"Churn rate: {churn_rate * 100:.2f}%")

# Churn rate by International Plan
print("Churn by International Plan:")
df.groupBy("Int'l Plan", "Churn").count().show()

# Churn rate by VMail Plan
print("Churn by VMail Plan:")
df.groupBy("VMail Plan", "Churn").count().show()

# Churn by State
print("Churn by State:")
df.groupBy("State", "Churn").count().orderBy("count", ascending=False).show()

# Average customer service calls for churned and non-churned customers
print("Average Customer Service Calls for Churned vs Non-Churned Customers:")
df.groupBy("Churn").agg(mean("CustServ Calls").alias("Avg CustServ Calls")).show()

# Churn by Area Code
print("Churn by Area Code:")
df.groupBy("Area Code", "Churn").count().orderBy("count", ascending=False).show()

# Average values for key metrics for churned vs non-churned customers
print("Key Metric Averages for Churned vs Non-Churned Customers:")
df.groupBy("Churn").agg(
    mean("Day Mins").alias("Avg Day Mins"),
    mean("Day Calls").alias("Avg Day Calls"),
    mean("Eve Mins").alias("Avg Eve Mins"),
    mean("Night Mins").alias("Avg Night Mins"),
    mean("Intl Mins").alias("Avg Intl Mins"),
    mean("CustServ Calls").alias("Avg CustServ Calls")
).show()

# Stop the Spark session
spark.stop()
