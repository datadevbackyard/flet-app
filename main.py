import flet as ft
from leaning_curve import LearningCurvePage
from evaluation_metrics import EvaluateMetricsPage
def main(page:ft.Page):


    def update_content(index):

        content_container.controls.clear()

        if index == 0:
            content_container.controls.extend(EvaluateMetricsPage(page))
        elif index == 1:
            content_container.controls.extend(LearningCurvePage())
        page.update()


    tabs = ft.Tabs(
        selected_index = 0,
        on_change = lambda e: update_content(e.control.selected_index),
        tabs = [
            ft.Tab(text = "Evaluation Metrics"),
            ft.Tab(text="Learning Curve")
        ]
    )

    content_container = ft.Column(
        controls= EvaluateMetricsPage(page),
        expand = True
    )



    
    page.add(
        ft.Column(
            controls = [
                tabs,
                content_container
            ],
            expand= True
        )
    )



ft.app(target=main)