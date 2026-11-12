"""Offline regression tests for the actual judgment parser."""
import ast
import importlib.util
import os
from pathlib import Path
import runpy
import subprocess
import sys
import unittest
from unittest.mock import patch

SOURCE = Path(os.environ.get("JANUS_PARSER", Path(__file__).resolve().parents[1] / "utils/output_parser.py"))
spec = importlib.util.spec_from_file_location("janus_parser", SOURCE)
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)
tree = ast.parse(SOURCE.read_text())
CASES = next(ast.literal_eval(node.value) for node in ast.walk(tree)
             if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "test_cases" for t in node.targets))


class ParserTests(unittest.TestCase):
    def test_malformed(self):
        for text in ("", "[RESULT]", "Score: unknown", "[x]", None, 3):
            with self.subTest(text=text):
                self.assertEqual(parser.parse_judgment_abs(text), (None, None))

    def test_ordinary_numbers(self):
        for text in ("There are 3 reasons and 2 examples.", "Version 2024 has 5 features.", "3", "(3)"):
            with self.subTest(text=text):
                self.assertEqual(parser.parse_judgment_abs(text), (None, None))

    def test_range_unchanged(self):
        self.assertEqual(parser.parse_judgment_abs("Score: 12"), ("Score: 12", 12))
        self.assertEqual(parser.parse_judgment_abs("Score: 0"), ("Score: 0", 0))

    def test_relative_unchanged(self):
        for text, expected in (("[[A]]", "A"), ("[[B]]", "B"), ("[[C]]", "tie"), ("none", "error")):
            self.assertEqual(parser.parse_judgment_rel(text), (text, expected))

    def test_self_check_success(self):
        p = subprocess.run([sys.executable, "-B", str(SOURCE)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0)
        self.assertIn("All tests passed!", p.stdout)

    def test_self_check_failure_nonzero(self):
        with patch("re.search", return_value=None):
            with self.assertRaises(SystemExit) as ctx:
                runpy.run_path(str(SOURCE), run_name="__main__")
        self.assertEqual(ctx.exception.code, 1)


def example_test(text, expected_score):
    def test(self):
        feedback, score = parser.parse_judgment_abs(text)
        self.assertEqual(score, expected_score)
        expected_feedback = text.split("[RESULT]")[0].strip() if "[RESULT]" in text else text
        self.assertEqual(feedback, expected_feedback)
    return test


for index, (text, score) in enumerate(CASES):
    setattr(ParserTests, f"test_embedded_{index:02d}", example_test(text, score))

if __name__ == "__main__":
    unittest.main()
