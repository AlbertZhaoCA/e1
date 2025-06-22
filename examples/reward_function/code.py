# import re
# from typing import Any, Dict, List
# from examples.reward_function.execute import run_scripts_with_chunk,worker


# def accuracy_reward(response: str, test_input: list, test_output: list) -> float:
#     if not response or not test_input or not test_output:
#         return 0.0

#     results = run_scripts_with_chunk(
#         code_list=[response] * len(test_input),
#         test_input_list=test_input,
#         time_limit_list=[1] * len(test_input),
#         worker=worker,
#         num_chunks=1
#     )
#     print(results)


#     correct_count = sum(1 for res, expected in zip(results, test_output) if res == expected)
#     return correct_count / len(test_output) if test_output else 0.0
    

# def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
#     if not isinstance(reward_inputs, list):
#         raise ValueError("Please use `reward_type=batch` for math reward function.")
    
#     scores = []
#     for reward_input in reward_inputs:
#         accuracy_score = accuracy_reward(reward_input["response"], reward_input["test_input"], reward_input["test_output"])
#         print(accuracy_score)
#         scores.append(
#             {
#                 "overall": accuracy_score,
#                 "accuracy": accuracy_score,
#             }
#         )

#     return scores

# from typing import Any, Dict, List
# from examples.reward_function.execute import run_scripts_with_timeout


# def accuracy_reward(response: str, test_input: list, test_output: list) -> float:
#     if not response or not test_input or not test_output:
#         return 0.0

#     results = run_scripts_with_timeout(
#         scripts=[response] * len(test_input),
#         inputs=test_input,
#         time_limit=0.1,
#     )


#     print(results)

#     correct_count = sum(1 for res, expected in zip(results, test_output) if res == expected)
#     return correct_count / len(test_output) if test_output else 0.0


# def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
#     if not isinstance(reward_inputs, list):
#         raise ValueError("Please use `reward_type=batch` for math reward function.")

#     print('compute score!')
#     scores = []
#     for reward_input in reward_inputs:
#         acc_score = accuracy_reward(
#             reward_input["response"],
#             reward_input["test_input"],
#             reward_input["test_output"]
#         )
#         print(f"score: {acc_score}")
#         scores.append({
#             "overall": acc_score,
#             "accuracy": acc_score,
#         })

#     return scores


# reward.py
from typing import Any, Dict, List
from examples.reward_function.lexecute import run_scripts_with_timeout, worker


def accuracy_reward(response: str, test_input: List[str], test_output: List[str]) -> float:
    if not response or not test_input or not test_output:
        return 0.0

    results = run_scripts_with_timeout(
        script=response,
        inputs=test_input,
        time_limit=0.1,
        worker=worker,
    )

    print("Execution results:", results)

    correct_count = sum(1 for res, expected in zip(results, test_output) if res.strip() == expected.strip())
    return correct_count / len(test_output)


def compute_score(reward_inputs: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    print('Computing scores...')
    scores = []
    for reward_input in reward_inputs:
        acc_score = accuracy_reward(
            reward_input["response"],
            reward_input["test_input"],
            reward_input["test_output"]
        )
        print(f"Score: {acc_score}")
        scores.append({
            "overall": acc_score,
            "accuracy": acc_score,
        })

    return scores
