from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Customer Churn Analysis") \
    .getOrCreate()

# Load data
csv_file_path = "/Users/anjaligupta/Desktop/project/project_1/pyspark/churn.csv"  # Use the absolute path
df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

# Show the first 5 rows
df.show(5)

# Stop the Spark session
spark.stop()
