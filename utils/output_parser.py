# Inspired by https://github.com/prometheus-eval/prometheus-eval/blob/main/eval/parser.py

import re

pattern = re.compile(
    r"""
    (?:                     # Non-capturing group for various result indicators
        \[RESULT\]          # [RESULT]
        |Score              # Score
        |\[SCORE\]          # [SCORE]
        |\[RESULT\]:        # [RESULT]:
        |Score:             # Score:
        |score:             # score:
        |Result:            # Result:
        |\[Result\]         # [Result]
        |score\s+of         # score of
        |Feedback:          # Feedback:
        |feedback:          # feedback:
        |\[Rating\]:
        |\[rating\]:
        |Rating:
        |rating:
    )
    \s*                     # Optional whitespace
    (?:                     # Non-capturing group for optional brackets or parentheses
        \(\s*               # Opening parenthesis with optional whitespace
        |\[\s*              # or opening square bracket with optional whitespace
        |                   # or nothing
    )
    \s*                     # Optional whitespace
    (\d+)                   # Capturing group for one or more digits
    """,
    re.IGNORECASE | re.VERBOSE,
)

# choice_pattern = re.compile(
#     r"""
#     (?:                     # Non-capturing group for various result indicators
#         \[Answer\]          
#         |Answer:             
#         |\[Choice\]          
#         |Choice:             
#         |\[Answer choice\]         
#         |Answer choice:        
#         |\[My choice\]         
#         |My choice:
#         |\*\*My choice\*\*:
#         |\*\*My choice:\*\*
#         |Thus,\s+my\s+choice\s+(?:is|would\s+be):
#     )
#     \s*                     # Optional whitespace
#     (?:                     # Non-capturing group for optional brackets or parentheses
#         \(\s*               # Opening parenthesis with optional whitespace
#         |\[\s*              # or opening square bracket with optional whitespace
#         |                   # or nothing
#     )
#     \s*                     # Optional whitespace
#     ((\*\*)?[A-D])             # Capturing group for A-D or one or more digits
#     \s*                     # Optional whitespace
#     (?:                     # Non-capturing group for optional closing brackets or parentheses
#         \s*\)               # Closing parenthesis with optional whitespace
#         |\s*\]              # or closing square bracket with optional whitespace
#         |                   # or nothing
#     )
#     \.?                     # Optional period after the choice
#     \s*                     # Optional whitespace
#     (.*)                    # Capture any text after the choice
#     """,
#     re.IGNORECASE | re.VERBOSE,
# )
choice_pattern = re.compile(r"(?:\[Answer\]|Answer:|\[Choice\]|Choice:|\[Answer choice\]|Answer choice:|\[My choice\]|My choice:|\*\*My choice\*\*:|\*\*My choice:\*\*|Thus,\s+my\s+choice\s+(?:is|would\s+be):)\s*(?:\(\s*|\[\s*|)\s*((?:\*\*)?[A-D](?:\*\*)?)\s*(?:\s*\)|\s*\]|)\.?\s*(.*)")


def parse_judgment_abs(judgment):
    matches = pattern.search(judgment)

    if matches:
        # Extract the first group that matches (ignoring None)
        result = next(int(match for match in matches.groups() if match), None)
        if result is not None:
            feedback = (
                judgment.split("[RESULT]")[0].strip()
                if "[RESULT]" in judgment
                else judgment
            )
            return feedback, result

    return None, None


def parse_judgment_alphabet(judgment):
    matches = choice_pattern.search(judgment)

    if matches:
        # Extract the first group that matches (ignoring None)
        result = next((match for match in matches.groups() if match), None)
        return judgment, result

    return judgment, None


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
    test_cases = [
        ("Answer explanation: Since the question asks about who didn't see the note and it mentions that both Ahmed Qureshi and Reem Safadi refused to sign a permission slip, implying they were either shown the note and didn't sign it or weren't shown it at all. Given the information, we cannot definitively determine whether either of them saw the note unless we're given more context on why they refused to sign it. However, considering the structure of the question, it appears to focus on action towards the note rather than seeing it. Therefore, the choices are skewed towards the reasoning behind their actions rather than their visibility towards the note itself. Hence, B. Unknown might be incorrect because the question presumes there's a context about seeing or not seeing the note directly related to their actions. Since the only action mentioned is refusing to sign the permission slip, which could be due to not seeing the note or for some other reason (like forgetting, misplacing the slip, etc.), we cannot accurately infer if it's a result of not seeing the note without more details. However, option C. Ahmed Qureshi did not sign the permission slip, suggesting a certain action taken (or not taken) regarding the note, which indirectly implies an interaction with the note, even if it's just by not signing. Based on the information provided and the interpretation of the question's intention, it seems like we're meant to determine the involvement or lack thereof in receiving or acknowledging the note directly related to their actions. My choice: A. Reem Safadi")
    ]

    def run_tests():
        failed_tests = []  # To keep track of failed tests

        for output, expected in test_cases:
            _, result = parse_judgment(output)
            if result != expected:
                failed_tests.append((output, expected, result))

        if failed_tests:
            print("Some tests failed:")
            for output, expected, result in failed_tests:
                print(f"  For input: '{output}', expected: {expected}, got: {result}")
        else:
            print("All tests passed!")

    run_tests()