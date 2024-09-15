from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("Data Cleaning and Transformation") \
    .getOrCreate()

# Load the CSV file
file_path = "/Users/anjaligupta/Desktop/project/project_1"
data = spark.read.csv(file_path, header=True, inferSchema=True)

# Display the schema
data.printSchema()

# Show initial data
data.show(5)


