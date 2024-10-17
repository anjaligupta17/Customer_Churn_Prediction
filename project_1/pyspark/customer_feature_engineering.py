from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, mean, when, corr
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Customer Churn Advanced Analysis") \
    .getOrCreate()

# Load data into a DataFrame
csv_file_path = "/Users/anjaligupta/Desktop/project/project_1/pyspark/churn.csv"
df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

# Show the first 5 rows
df.show(5)

# 1. Feature Engineering: Creating total call minutes and total charges features
df = df.withColumn("Total Mins", col("Day Mins") + col("Eve Mins") + col("Night Mins") + col("Intl Mins"))
df = df.withColumn("Total Charges", col("Day Charge") + col("Eve Charge") + col("Night Charge") + col("Intl Charge"))

# 2. Categorizing Customers as High/Low Usage based on Total Minutes
df = df.withColumn("Usage Category", when(col("Total Mins") > 500, "High").otherwise("Low"))

# Show new columns
df.select("Total Mins", "Total Charges", "Usage Category").show(5)

# 3. Correlation Analysis: Check correlation between numerical features and churn
numerical_features = ["Account Length", "Day Mins", "Eve Mins", "Night Mins", "Intl Mins", "CustServ Calls", "Total Charges"]
for feature in numerical_features:
    corr_value = df.select(corr(feature, "Churn").alias("correlation")).collect()[0][0]
    print(f"Correlation between {feature} and Churn: {corr_value:.4f}")

# 4. Churn by Usage Category (High vs. Low)
df.groupBy("Usage Category", "Churn").count().show()

# 5. Churn by Customer Service Calls: Bucket customer service calls
df = df.withColumn("CustServ Calls Category", when(col("CustServ Calls") >= 4, "High").otherwise("Low"))

df.groupBy("CustServ Calls Category", "Churn").count().show()

# 6. Simple Logistic Regression Model to Predict Churn

# Convert the Churn column from string (True/False) to integer (1/0)
df = df.withColumn("Churn", when(col("Churn") == "True.", 1).otherwise(0))

# Select features for logistic regression
feature_cols = ["Account Length", "CustServ Calls", "Day Mins", "Eve Mins", "Night Mins", "Intl Mins", "Total Charges"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_model = assembler.transform(df)

# Split data into training and testing sets
train_data, test_data = df_model.randomSplit([0.7, 0.3], seed=42)

# Initialize logistic regression
lr = LogisticRegression(labelCol="Churn", featuresCol="features")

# Train the model
lr_model = lr.fit(train_data)

# Predict on test data
predictions = lr_model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="Churn")
auc = evaluator.evaluate(predictions)
print(f"Area Under the ROC Curve (AUC): {auc:.4f}")

# Stop the Spark session
spark.stop()
