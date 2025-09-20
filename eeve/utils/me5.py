def get_detailed_instruct_for_ds_example(example: dict, task_description: str, source_column, target_column) -> dict:
    return {
        source_column: f'Instruct: {task_description}\nQuery: {example[source_column]}',
        # target_column: f'Instruct: {task_description}\nQuery: {example[target_column]}'
        target_column: example[target_column]
    }