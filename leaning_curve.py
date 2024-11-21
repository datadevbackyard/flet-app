import flet as ft
from flet.matplotlib_chart import MatplotlibChart
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

def LearningCurvePage():

    data = load_breast_cancer()
    X, y = data.data, data.target

    # RandomForestClassifier as the model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Generate learning curve data
    train_sizes, train_scores, test_scores = learning_curve(
        rf, X, y, cv=5, scoring="accuracy", train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0]
    )

    # Calculate mean and standard deviation
    train_scores_mean = train_scores.mean(axis=1)
    test_scores_mean = test_scores.mean(axis=1)

     # Plot the learning curve
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot(train_sizes, train_scores_mean, "o-", label="Training Score", color="blue")

    ax.plot(train_sizes, test_scores_mean, "o-", label="Cross-Validation Score", color="orange")
    ax.set_title("Learning Curve")
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="best")
    ax.grid(True)


    # Return the layout for this page
    return [
        ft.Text("Learning Curve", size=24, weight="bold"),
        ft.Text(
            "The learning curve shows the training and cross-validation accuracy as the training size increases.",
            selectable=True,
        ),
        MatplotlibChart(fig, expand=False, original_size= False),  # Add Matplotlib figure to the page
    ]