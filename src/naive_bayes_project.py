
"""
AI Programming Project – Naive Bayes Classifier
University of North Dakota – CSCI 384 AI Course | Spring 2025

Title: Predicting Hit Songs Using Naive Bayes
Total Points: 100 (+10 bonus points)

This is the main assignment script. You must complete each step where "YOUR CODE HERE" is indicated.
Use the provided helper modules (dataset_utils.py and naive_bayes_model.py) to assist you.
The NaiveBayesContinuous model is based on Artificial Intelligence: A Modern Approach, 4th US edition. 
GitHub repository: https://github.com/aimacode/aima-python
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
import dataset_utils as dataset_U
import naive_bayes_model as NB_model

# ---------------------------------------------------------------
# STEP 1 [10 pts]: Load the Dataset
# ---------------------------------------------------------------
data = dataset_U.load_dataset("../data/spotify_hits.csv")
# loding the data set
print(f"Data shape: {data.shape}")
print(data.head())

# ---------------------------------------------------------------
# STEP 2 [10 pts]: Create a Binary Target Column
# ---------------------------------------------------------------
# - Create a new column 'hit' from 'popularity'. A song is a hit if popularity ≥ 70; otherwise, it is not a hit.
data['hit'] = data['popularity'].apply(lambda x: 1 if x >= 70 else 0)

# - Delete the original 'popularity' column as it is no longer needed.
data = data.drop(columns =['popularity'])

# - Display unique values of the 'hit' column to verify the transformation.
print(data.head(), "\nSong is a 'hit' if hit = 1, not hit = 0.")


# ---------------------------------------------------------------
# STEP 3 [10 pts]: Preprocess the Dataset
# ---------------------------------------------------------------
# - Prepare the dataset for training by:
#   1. Keeping only numeric columns.
#   2. Removing rows with missing values.
# - Ensure that the 'hit' column is included in the final DataFrame for target variable.

# Select numeric columns and remove missing values.
data = data.select_dtypes(include='number').dropna()

print(f"data shape: {data.shape}") 

# ---------------------------------------------------------------
# STEP 4 [10 pts]: Train/Test Split
# ---------------------------------------------------------------
# - Split the dataset into training (80%) and testing (20%) sets.

# Split the dataset.
train_df, test_df = dataset_U.split_dataset(data, 'hit', test_size=0.2)

# Display the shape of the training and testing DataFrames.
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# ---------------------------------------------------------------
# STEP 5 [20 pts]: Train the Naive Bayes Model
# ---------------------------------------------------------------
# - Wrap the training DataFrame using the DataSet class (from dataset_utils.py) with 'hit' as the target.
# - Train the NaiveBayesContinuous model (from naive_bayes_model.py) using the training DataSet.

# Create the DataSet object and train the model. Don't forget to import the NaiveBayesContinuous class.
train_dataset = dataset_U.DataSet(train_df, target='hit')
model = NB_model.NaiveBayesContinuous(train_dataset) 

# ---------------------------------------------------------------
# STEP 6 [20 pts]: Make Predictions and Evaluate
# ---------------------------------------------------------------
# - For each song in the test set, extract its features (all columns except 'hit') as a dictionary.
# - Use the trained model to predict the label.
# - Compare the prediction to the true 'hit' value and compute the overall accuracy.

# Write your loop to predict and calculate accuracy.
correct = 0
total = 0
for _, row in test_df.iterrows():
    features = row.drop(labels='hit').to_dict()
    true_label = row['hit']
    predicted_label = model(features)
    if predicted_label == true_label:
        correct += 1
    total+=1
accuracy = correct/total
print(f"Model accuracy: {accuracy:.2f}")


# ---------------------------------------------------------------
# STEP 7 [20 pts]: Answer Conceptual Questions
# ---------------------------------------------------------------
# For each question, assign your answer (as a capital letter "A", "B", "C", or "D"). Add explanations for your choices.

# Q1 [10 pts]: Which features are most likely to influence whether a song is a hit? Explain your reasoning.
#   A. track_id and duration_ms
#   B. danceability, acousticness, and instrumentalness
#   C. popularity and tempo
#   D. artist name and genre

q1_answer = "B"  # YOUR ANSWER HERE
q1_explanation = "These traits are more predictive of popularity compared to categorical attributes
like artist name or genre, which are not numeric and appear inconsistently."

# Hint: Correlation analysis can help identify influential features. Sort descending by correlation with the target variable. Target variable has correlation of 1.0.

# YOUR CODE HERE:
correlations = data.corr()['hit'].drop('hit')
print("Correlation of features with 'hit':")
print(correlations.sort_values(ascending=False))

# Q2 [5 pts]: What assumption does the Naive Bayes model make about the input features? Explain your reasoning.
#   A. They follow a uniform distribution.
#   B. They are normally distributed.
#   C. They are independent given the target class.
#   D. They are weighted by importance.
# Hint: Refer to the Naive Bayes assumption. Ref: https://en.wikipedia.org/wiki/Naive_Bayes_classifier

q2_answer = "C"  
q2_explanation = "The assumption made by the Naive Bayes model is that its independent given the target class, shown in our code the target class being 'hit'." 

# Q3 [5 pts]: What is a likely difference if a decision tree is used instead of Naive Bayes? Explain your reasoning.
#   A. The model will assume independence of features.
#   B. The model will assign probabilities instead of decision rules.
#   C. The model will create splits based on feature thresholds.
#   D. The model will always perform worse.
# Hint: Consider how decision trees work compared to Naive Bayes. Ref: https://en.wikipedia.org/wiki/Decision_tree_learning

q3_answer = "C"  
q3_explanation = "Decision trees create branches based on conditions so it would create hard splits." 

'''
# ---------------------------------------------------------------
# BONUS SECTION: Advanced Analysis [10 bonus pts]
# ---------------------------------------------------------------
# BONUS Task 1 [6 pts]:
# - A. Compute and display a confusion matrix comparing the true labels to your model's predictions.
# - B. Interpret the confusion matrix. What does it tell you about the model's performance?
# - Hint: You may use sklearn.metrics.confusion_matrix. Ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

# - A. Compute and display a confusion matrix comparing the true labels to your model's predictions.
from sklearn.metrics import confusion_matrix
# YOUR CODE HERE:
y_true = # YOUR CODE HERE
y_pred = # YOUR CODE HERE
cm = # YOUR CODE HERE
print("Confusion Matrix:")
print(cm)

# - B. Interpret the confusion matrix. What does it tell you about the model's performance?
bonus_task_1_interpretation = "" # YOUR INTERPRETATION HERE

# BONUS Task 2 [4 pts]:
# - Experiment with different thresholds for defining a hit (thresholds = [65, 70, 75, 80]).
# - Determine which threshold gives the best model accuracy.
# - Report your best threshold and the corresponding accuracy.

# - Hint: 
# Try iterating over a list of possible thresholds (for example, 65, 70, 75, 80). For each threshold, update your target column 'hit' so that a song is marked as a hit if its popularity is greater than or equal to that threshold. Then, split the dataset, train your model, and compute its accuracy on the test set. Store each threshold's accuracy (for example, in a dictionary), and finally, select the threshold with the highest accuracy. Assign this best threshold and its accuracy to the variables best_threshold and best_accuracy.

# - Note: DO NOT write your code here. Only provide the best threshold and accuracy below.
best_threshold = 0  # YOUR BEST THRESHOLD HERE
best_accuracy = 0.0  # YOUR BEST ACCURACY HERE (update after testing)
print(f"Best threshold: {best_threshold}, Best accuracy: {best_accuracy:.2f}")
