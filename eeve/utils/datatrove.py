import dataclasses
from datatrove.data import Document
from huggingface_hub import hf_hub_download


def _reader_adapter_with_column_info(self, data: dict, path: str, id_in_file: int | str):
    """
    Modified version of the default adapter to adapt input data into the datatrove Document format
    Differs by adding source dataset column names information to metadata.

    Args:
        data: a dictionary with the "raw" representation of the data
        path: file path or source for this sample
        id_in_file: its id in this particular file or source

    Returns: a dictionary with text, id, media and metadata fields

    """
    dataset_info = {
        "default_columns": list(data.keys()),
        "rename_column": {'text': self.text_key} 
    }
    metadata = data.pop("metadata", {})
    if isinstance(metadata, str):
        import json

        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            pass
    if not isinstance(metadata, dict):
        metadata = {"metadata": metadata}
    return {
        "text": data.pop(self.text_key, ""),
        "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
        "media": data.pop("media", []),
        "metadata": metadata | data | dataset_info,  # remaining data goes into metadata
    }


def _writer_adapter_with_column_restore(self, document: Document) -> dict: 
    """
    You can create your own adapter that returns a dictionary in your preferred format
    Args:
        document: document to format

    Returns: a dictionary to write to disk

    """
    data = {key: val for key, val in dataclasses.asdict(document).items() if val}
    if "metadata" in data:
        metadata = data.pop("metadata")
        
        default_columns = metadata.pop("default_columns", None)
        rename_column = metadata.pop("rename_column", None)
        
        data.update(metadata)
        
        if rename_column:
            for old_name, new_name in rename_column.items():
                if old_name in data and old_name != new_name:
                    data[new_name] = data.pop(old_name)
        
        if default_columns:
            data = {key: data[key] for key in default_columns if key in data}
    return data


def fasttext_model_get_path(
    local_model_path: str | None = None,
    hf_repo_name: str | None = None,
    filename: str | None = None
) -> str:   
    if hf_repo_name:
        if not filename:
            raise ValueError("If `repo_id` is specified, `filename` must also be provided.") 
        else:
            return hf_hub_download(repo_id=hf_repo_name, filename=filename)
    if not hf_repo_name and not local_model_path:
        raise ValueError("А че качать-то")
    return local_model_path