from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

import faiss
import numpy as np
from tqdm.notebook import tqdm


# https://github.com/VikhrModels/effective_llm_alignment/blob/main/src/utils/embeddings_utils.py#L136
def _faiss_deduplicate_single_v3(
    embeddings: np.ndarray,
    similarity_threshold=0.9,
    index_class=faiss.IndexFlatIP,
    index_args=[],
):
    # Initialize FAISS index (with inner product similarity by default)
    index = index_class(embeddings.shape[1], *index_args)

    # Add embeddings to the index
    index.add(embeddings)

    # Perform range search to find all neighbors within a similarity threshold
    result = index.range_search(embeddings, similarity_threshold)

    # Extract result components: lims indicate result ranges per query, D is distances, I are indices
    lims, _, I = result

    # Initialize a set of indices to keep and a set for visited indices
    keep = np.ones(len(embeddings), dtype=bool)
    visited = np.zeros(len(embeddings), dtype=bool)

    # Process the results of the range search to deduplicate embeddings
    for i in range(len(embeddings)):
        if visited[i]:  # If already handled, continue
            continue

        # Get the start and end of the neighbors of the i-th query from lims
        start_idx, end_idx = lims[i], lims[i + 1]
        neighbors = I[start_idx:end_idx]

        # Exclude the embedding itself (distance 0 or self-index i)
        neighbors = neighbors[neighbors != i]

        # Mark visited for all neighbors
        visited[neighbors] = True

        # Keep only the current embedding (i-th) and mark rest as duplicates
        keep[neighbors] = False

    # Return only the unique embeddings based on the `keep` array
    unique_indices = np.where(keep)[0]
    return embeddings[unique_indices], unique_indices


def faiss_deduplicate_mr(
    embeddings: np.ndarray,
    max_workers=cpu_count(),
    batch_size=100_000,
    similarity_threshold=0.9,
    index_class=faiss.IndexFlatIP,
    index_args=[],
):
    num_embeddings = embeddings.shape[0]
    batch_starts = list(range(0, num_embeddings, batch_size))

    # Создаем список батчей, которые нужно обработать
    batches = [
        embeddings[start : min(start + batch_size, num_embeddings)]
        for start in batch_starts
    ]

    all_unique_embeddings = []
    all_unique_indices = []

    # Используем ThreadPoolExecutor для параллельного выполнения задач
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_start = {
            executor.submit(
                _faiss_deduplicate_single_v3,
                batch,
                similarity_threshold,
                index_class,
                index_args,
            ): start
            for start, batch in zip(batch_starts, batches)
        }

        # tqdm для отслеживания выполнения параллельных задач
        for future in tqdm(
            as_completed(future_to_start),
            total=len(future_to_start),
            desc="Processing batches",
            unit="batch",
        ):
            batch_start = future_to_start[future]
            unique_embeddings, unique_indices = future.result()

            # Shift the local indices by the starting index of the batch to map back to global indices
            unique_indices_global = unique_indices + batch_start

            all_unique_embeddings.append(unique_embeddings)
            all_unique_indices.append(unique_indices_global)

    # Объединяем результаты
    # all_unique_embeddings = np.vstack(all_unique_embeddings)
    all_unique_indices = np.concatenate(all_unique_indices)

    return embeddings[all_unique_indices], all_unique_indices


def faiss_deduplicate_two_pass(
    embeddings: np.ndarray,
    max_workers=cpu_count(),
    batch_size=100_000,
    similarity_threshold=0.9,
    index_class=faiss.IndexFlatIP,
    index_args=[],
):
    num_embeddings = embeddings.shape[0]
    batch_starts = list(range(0, num_embeddings, batch_size))

    batches = [
        embeddings[start : min(start + batch_size, num_embeddings)]
        for start in batch_starts
    ]

    results_by_start = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_start = {
            executor.submit(
                _faiss_deduplicate_single_v3,
                batch,
                similarity_threshold,
                index_class,
                index_args,
            ): start
            for start, batch in zip(batch_starts, batches)
        }

        for future in tqdm(
            as_completed(future_to_start),
            total=len(future_to_start),
            desc="Processing batches",
            unit="batch",
        ):
            batch_start = future_to_start[future]
            unique_embeddings, unique_indices = future.result()

            unique_indices_global = unique_indices + batch_start
            results_by_start[batch_start] = (unique_embeddings, unique_indices_global)

    ordered = [results_by_start[start] for start in sorted(results_by_start)]

    if not ordered:
        raise ValueError(
            "Empty input: embeddings has shape (0, d); nothing to deduplicate"
        )

    all_unique_embeddings = [ue for ue, _ in ordered]
    all_unique_indices = [ui for _, ui in ordered]

    uniques_stage1 = np.vstack(all_unique_embeddings)
    indices_stage1 = np.concatenate(all_unique_indices)

    if len(batch_starts) > 1:
        _, keep_indices_local = _faiss_deduplicate_single_v3(
            uniques_stage1,
            similarity_threshold,
            index_class,
            index_args,
        )
        final_indices = indices_stage1[keep_indices_local]

        return embeddings[final_indices], final_indices
    return embeddings[indices_stage1], indices_stage1
