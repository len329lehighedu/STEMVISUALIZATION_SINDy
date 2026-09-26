import unittest

from bokeh.models import Plot, Tabs

from flask_app import app as flask_app
from main import build_app


class AppSmokeTests(unittest.TestCase):
    def test_build_app_wires_all_four_tabs(self):
        app = build_app()
        self.assertIsInstance(app, Tabs)
        self.assertEqual(
            [panel.title for panel in app.tabs],
            ["🏋 Train & Validate", "🧪 Test", "🔮 Predict", "🎲 Ensemble"],
        )
        self.assertTrue(all(plot.renderers for plot in app.select(
            {"type": Plot})))

    def test_questions_fragment_exposes_local_assistant_knowledge_base(self):
        client = flask_app.test_client()
        response = client.get("/fragment/questions")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("data-sindy-assistant", html)
        self.assertIn("Local knowledge base", html)
        question_count = html.count('data-question-id=')
        answer_count = html.count('class="assistant-kb-entry"')
        self.assertGreater(question_count, 0)
        self.assertEqual(question_count, answer_count)

    def test_index_loads_assistant_javascript(self):
        client = flask_app.test_client()
        response = client.get("/static/js/sindy-assistant.js")
        self.assertEqual(response.status_code, 200)
        javascript = response.get_data(as_text=True)
        self.assertIn("window.initSindyAssistant", javascript)
        response.close()


if __name__ == "__main__":
    unittest.main()
