Predicting Video Game Sales Using Linear Regression and Classification

This project is a part of the AAI-501 course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

-- Project Status: Completed

Installation

Primary repository project files:

README.md – file containing project description/details
data\updated.py – Python code developed for an end-to-end linear regression and classification predictions.
vgchartz_2024 – Video game data from Kaggle.
LICENSE.docx and .md – The MIT License to share the project
To use this project, you need to clone the project repository, use any python IDE to modify and execute the code in data\updated.py:

git init

git clone https://github.com/lex-lim/AAI-501-Final.git

Project Intro/Objective

This project involves performing exploratory data analysis, feature engineering, and model building using linear regression and decision tree classification to predict video game sales and classify their sales impact, respectively.

Partner(s)/Contributor(s)

Alexis Lim, Olga Pospelova
Team 4
Methods Used

Linear Regression Model
Decision Tree Calssification
Descriptive Statistics
Histograms
Exploratory Data Analysis (EDA)
Data Manipulation/Mapping
PEP 8
APA 7

Python v3.11.7
GitHub
Slack
Spyder IDE v5.5.1
Google Docs
Canvas
Jupyter
Google Colab
MS Word
MS PowerPoint
Zoom
Markdown
Project Description

This project involves the development of a linear regression model and a decision tree classifier to predict video game sales and classify their sales impact based on various features such as console, genre, publisher, and critic scores.

Dataset used: Video Game Sales dataset from VGChartz located on Kaggle.

The dataset includes:

Game Details: console, genre, publisher, developer, critic score.
Sales Data: total sales, NA sales, JP sales, PAL sales, other sales.
Release Information: release year, release month.
12 total columns/features.
(https://www.kaggle.com/datasets/asaniczka/video-game-sales-2024/data)

Data Dictionary:
Console: Various gaming consoles (e.g., PS3, PS4, X360).
Genre: Types of video games (e.g., Action, Shooter).
Publisher: Companies that published the games (e.g., Rockstar Games, Activision).
Developer: Companies that developed the games (e.g., Rockstar North, Treyarch).
Critic Score: The average score given by critics.
Sales Data: Sales in millions in different regions.
Release Year: The year the game was released.
Release Month: The month the game was released.
Dataset Size: contains data for 1885 respondents
Data Size: Contains data for 64,016 video games.

Question: Can we predict the total sales of video games based on their features? Additionally, can we classify the sales impact of video games as tentpole, moderate, or minor?

EDA Analysis:
Initial Data Inspection: Displaying the first few rows, data structure, and statistical summary.
Missing Values Handling: Converting date columns, filling missing values for numeric columns.
Feature Engineering: Creating new columns for release year and month, and dropping unnecessary columns.
Label Encoding: Converting categorical columns into numeric format using LabelEncoder.
Linear Regression Model:

Features and Target Variable: Using features like console, genre, publisher, developer, critic score, release year, and release month to predict total sales.
Model Training and Evaluation: Splitting the data into training and testing sets, training the model, and evaluating using Mean Squared Error (MSE) and R-squared.
Decision Tree Classifier:

Sales Impact Categorization: Defining thresholds for tentpole, moderate, and minor impact based on total sales quantiles.
Model Training and Evaluation: Splitting the data into training and testing sets, training the classifier, and evaluating using accuracy, precision, recall, and confusion matrix.
Summary of Methods:

Data Loading and Preprocessing:
Handling missing values and data conversion.
Dropping unnecessary columns and label encoding.

Exploratory Data Analysis (EDA):
Inspecting data structure and statistical summary.

Linear Regression:
Splitting data into training and testing sets.
Training the model and evaluating performance.

Decision Tree Classification:
Defining and categorizing sales impact.
Training the classifier and evaluating performance.

Outcome:
Predicted total sales with a Mean Squared Error of 0.03 and an R-squared of 0.82.
Classified sales impact with an accuracy of 0.94, providing insights into the factors influencing high-impact game sales.
This comprehensive approach aims to enhance the understanding of video game sales dynamics and provides a robust framework for predicting sales and categorizing their impact.
