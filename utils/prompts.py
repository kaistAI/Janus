# ---------------------------------------------------------------------------- #
#                                Absolute                                      #
# ---------------------------------------------------------------------------- #


ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

ABS_USER_PROMPT_TEMPLATE = """
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
{score_rubric}

###Feedback: """


# ---------------------------------------------------------------------------- #
#                               Rubric generation                              #
# ---------------------------------------------------------------------------- #

SYSTEM_PROMPT_RUBRIC_GENERATION_SINGLE_PREFERENCE = """You are helpful and creative rubric generator. You should brainstorm a creative and impressive rubric to evaluate how well a language model follows the given preference.
You are given 3 example rubrics and a preference for a specific dimension.
Suppose you were to evaluate a language model. Create a score rubric that can assess whether the model has generated a response that is tailored to the preference.

[Rules]
- The rubric should be structured to evaluate whether a language model has created a response considering the given preferences.
- Please do not generate any other opening, closing, and explanations.
- Output the score rubric in JSON format. Please format the output as same as the examples with no extra or surrounding text."""

USER_PROMPT_TEMPLATE_RUBRIC_GENERATION_SINGLE_PREFERENCE = """
# Example rubrics
{example_1}

{example_2}

{example_3}

# Preference
{preference}

# Generated rubric:
"""

# ---------------------------------------------------------------------------- #
#                              Sampling parameters                             #
# ---------------------------------------------------------------------------- #

# Open source LLM inference parameters
SAMPLING_PARAMS = {
    "max_tokens": 1024,
    "temperature": 1.0,
    "top_p": 0.9,
    "repetition_penalty": 1.03,
}

# OpenAI inference parameters
SAMPLING_PARAMS_OPENAI = {"max_tokens": 1024, "temperature": 1.0, "top_p": 0.9}

# OpenAI rubric generation parameters
SAMPLING_PARAMS_OPENAI_RUBRIC_GENERATION = {
    "max_tokens": 1024,
    "temperature": 1.0,
    "top_p": 0.9,
    "response_format": {"type": "json_object"},
}
