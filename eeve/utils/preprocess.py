def get_detailed_instruct_for_one_sample(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


# def get_detailed_instruct_for_ds_example(example: dict, task_description: str) -> str:
#     return {
#         k: get_detailed_instruct_for_one_sample(query=v, task_description=task_description)
#         for k, v in example.items()
#     }