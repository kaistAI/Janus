import re


def parse_judgment_rel(judgment):
    if "[[A]]" in judgment:
        winner = "A"
    elif "[[B]]" in judgment:
        winner = "B"
    elif "[[C]]" in judgment:
        winner = "tie"
    else:
        winner = "error"
    return judgment, winner


abs_pattern = r"""(?:\[RESULT\]|Score|\[SCORE\]|\[RESULT\]:|Score:|score:|Result:|\[Result\]|score of)\s*(?:\(\s*|\[\s*|)\s*(\d+)"""


def parse_judgment_abs(output):
    # Extended pattern to match more variations of result presentation
    # pattern = r"""
    #     (?:\[RESULT\]|Score:|score:|Result:|\[Result\]|score of|\(|\[|\])\s*  # Match different prefixes including '[RESULT]', 'Score:', etc.
    #     (?:\[RESULT\]|Score:|score:|Result:|\[Result\]|score of|\(|\[|\])\s*
    #     |(\d+)\s*                               # Catch trailing numbers
    #     |\((\d+)\)                              # Catch numbers within parentheses
    #     |\[(\d+)\]                              # Catch numbers within brackets
    # """

    if not isinstance(output, (str, bytes)):
        print(f"Invalid type for output: {type(output)}")
        return None, None

    matches = re.search(abs_pattern, output, re.IGNORECASE | re.VERBOSE)

    if matches:
        # Extract the first group that matches (ignoring None)
        result = next((int(match) for match in matches.groups() if match), None)
        if result is not None:
            feedback = (
                output.split("[RESULT]")[0].strip() if "[RESULT]" in output else output
            )
            return feedback, result

    return None, None


if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Absolute mode test cases (a2a, a2r)
        ("Good job. [RESULT] 3", 3),
        ("Needs improvement. [RESULT] Score: 2", 2),
        ("Well done. [RESULT] Result: 4", 4),
        ("Average. [RESULT] 4/5", 4),
        ("Excellent. [RESULT] 5 out of 5", 5),
        ("Poor performance. [RESULT] score of 1", 1),
        ("Good job. [Result] 3", 3),
        ("Needs improvement. [Result] Score: 2", 2),
        ("Well done. [Result] Result: 4", 4),
        ("Average. [Result] 4/5", 4),
        ("Excellent. [Result] 5 out of 5", 5),
        ("Poor performance. [Result] score of 1", 1),
        ("Good job. [3]", 3),
        ("Good job. (Score 5)", 5),
        ("Good job. [Score 4]", 4),
        ("Good job. score: 3", 3),
        ("Good job. Score: 3", 3),
        ("Good job. score of 1", 1),
        ("Good job. [RESULT] (5)", 5),
    ]

    def run_tests():
        failed_tests = []  # To keep track of failed tests

        for output, expected in test_cases:
            _, result = parse_judgment_abs(output)
            if result != expected:
                failed_tests.append((output, expected, result))

        if failed_tests:
            print("Some tests failed:")
            for output, expected, result in failed_tests:
                print(f"  For input: '{output}', expected: {expected}, got: {result}")
        else:
            print("All tests passed!")

    run_tests()
