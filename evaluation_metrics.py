import flet as ft
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def EvaluateMetricsPage(page):

    # Load Breast Cancer dataset
    data = load_breast_cancer()

    X = data.data
    y = data.target

    # Split the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Extract TP, TN, FP, FN values
    TN, FP, FN, TP = cm.ravel()

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    accuracy_color = "black" if accuracy > 0.90 else "orange" if accuracy > 0.80 else "red"
    precision_color = "black" if precision > 0.90 else "orange" if precision > 0.80 else "red"
    recall_color = "black" if recall > 0.90 else "orange" if recall > 0.80 else "red"
    f1_color = "black" if f1 > 0.90 else "orange" if f1 > 0.80 else "red"

    left_column = ft.Column(
        controls=[
            ft.Text("Confusion Matrix", size=24, weight="bold", color="blue"),
            ft.Divider(),
            ft.Row(
                controls=[ft.Text("Actual Values", size=20, weight="bold")],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            ft.Row(
                    controls=[
                        # Column for the vertical text
                        ft.Column(
                            controls=[
                                ft.Text(
                                    "Predicted Values",
                                    size=20,
                                    weight="bold",
                                    rotate=ft.Rotate(-89.5),  # Rotate the text vertically
                                    
                                )
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,  # Center the text vertically
                        
                        
                        ),
            # Column for the grid
                    ft.Column(
                        controls=[
                            ft.GridView(
                                runs_count=2,  # Number of rows
                                spacing=10,  # Space between cells
                                controls=[
                                    ft.Container(
                                        content=ft.Text("TP", size=20, color="black"),
                                        bgcolor="orange",
                                        alignment=ft.alignment.center,
                                        expand=True,
                                        width= page.width
                                    ),
                                    ft.Container(
                                        content=ft.Text("FP", size=20, color="black"),
                                        bgcolor="orange",
                                        alignment=ft.alignment.center,
                                        expand=True,
                                        width= page.width
                                    ),
                                    ft.Container(
                                        content=ft.Text("FN", size=20, color="black"),
                                        bgcolor="lightgreen",
                                        alignment=ft.alignment.center,
                                        expand=True,
                                        width= page.width
                                    ),
                                    ft.Container(
                                        content=ft.Text("TN", size=20, color="black"),
                                        bgcolor="lightgreen",
                                        alignment=ft.alignment.center,
                                        expand=True,
                                        width= page.width
                                    ),
                                ]
                            )
                        ],
                expand=True,  # Allow the grid column to take up the remaining space
                alignment=ft.alignment.center,
            ),
                ],
                expand=True,
            )

        ],
        expand=True,  # Let the column expand to fit available space
    )

    
    # Right column with evaluation metrics
    right_column = ft.Column(
        controls=[
            ft.Text("Model Evaluation", size=24, weight="bold", color="blue"),
            ft.Divider(),
            # Accuracy container
            ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Text("Accuracy Formula:"),
                        ft.Text("Accuracy = (TP + TN) / (TP + TN + FP + FN)"),
                        ft.Text(f"Accuracy = {accuracy:.2f}", size=18, color=accuracy_color),
                    ]
                ),
                bgcolor="lightblue",
                border_radius=5,
                padding=10,
                expand = True,
                width = page.width,
                
            ),
            # Precision container
            ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Text("Precision Formula:"),
                        ft.Text("Precision = TP / (TP + FP)"),
                        ft.Text(f"Precision = {precision:.2f}", size=18, color=precision_color),
                    ]
                ),
                bgcolor="lightblue",
                border_radius=5,
                padding=10,
                expand = True,
                width = page.width,
                
            ),
            # Recall container
            ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Text("Recall Formula:"),
                        ft.Text("Recall = TP / (TP + FN)"),
                        ft.Text(f"Recall = {recall:.2f}", size=18, color=recall_color),
                    ]
                ),
                bgcolor="lightblue",
                border_radius=5,
                padding=10,
                expand = True,
                width = page.width,
                
            ),
            # F1 score container
            ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Text("F1 Score Formula:"),
                        ft.Text("F1 = 2 * (Precision * Recall) / (Precision + Recall)"),
                        ft.Text(f"F1 Score = {f1:.2f}", size=18, color=f1_color),
                    ]
                ),
                bgcolor="lightblue",
                border_radius=5,
                padding=10,
                expand = True,
                width = page.width,
                
            ),
        ],
        expand=True,
    )


    # Add row with left and right columns
    return [
        ft.Row(
            controls=[left_column, right_column],
            spacing=20,
            expand=True,
        )]