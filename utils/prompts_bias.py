multiple_choice_system_prompts = [
    "Read the given context carefully and choose the appropriate answer from A, B, C for the question. Respond with a single alphabet letter.",
    
    "Read the following passage and choose the correct answer from options A, B, C for the question. Answer with only one alphabet letter without explanation.",
    
    "Considering the given information, choose the most appropriate answer from options A, B, C for the question. Answer with only one alphabet letter without explanation.",
    
    "Read the given context and select the correct answer from options A, B, C. Answer only with the alphabet letter without explanation.",

    "Read the following and answer with only one alphabet letter from the given options."
]

multiple_choice_prompt = """Context: {context}
    Question: {question}
    A: {a}
    B: {b}
    C: {c}
    Answer: """