import os
import re
import copy
import math



def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    prompts = kwargs.get("prompts")
    print(f"solution are {solution}, completion are {contents}")
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else content.strip()

            if student_answer.lower() == ground_truth.lower():
                reward = 1.0
        except Exception:
            if content.lower() == sol.lower():
                reward = 1.0

        rewards.append(reward)
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>"
    completion_contents = [completion[0]["content"] for completion in completions]

    rewards = []
    for content in completion_contents:
        if re.fullmatch(r"<think>.*?</think>\s*<answer>.*?</answer>", content, re.DOTALL):
            rewards.append(0.7)
        elif re.search(pattern, content, re.DOTALL):
            rewards.append(0.0)
        else:
            rewards.append(-0.7)

    return rewards


