# General system prompt copied from open-r1
GENERAL_SYSTEM_PROMPT = (
    'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant '
    'first thinks about the reasoning process in the mind and then provides the user with the answer in a word or a phrase as required. You should output the reasoning process firstly and then output the answer.The reasoning '
    'process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., '
    '<think> reasoning process here </think><answer> answer here </answer>'
)

# General question prompt copied from R1-V
GENERAL_QUESTION_PROMPT = '{Question}. First do reasoning and give the answer as required. Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.'

REASONING_CONFIDENCE_SYSTEM_PROMPT = (
    'You are an assistant for estimating the reasoning confidence. The User gives a image with a question, and his own reasoning process, you should evaluate the reasoning is correct, '
    'giving the corresponding option directly without any extra text.'
)
REASONING_CONFIDENCE_PROMPT = ("""Answer whether the **reasoning** is correct for the given **question**. 

**Question:** {question}

**reasoning:** {reasoning}

Is the **reasoning** correct?
A) True
B) False

The **reasoning** is: [A / B], depending whether the **reasoning** is correct given the **question** with the image.

Only output A or B in one letter, without any extra text.
"""
)

