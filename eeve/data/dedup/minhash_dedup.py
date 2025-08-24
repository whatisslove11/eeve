import os
import yaml
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.tokens import TokensCounter
from datatrove.utils.hashing import HashConfig

from eeve.utils.datatrove import create_reader_and_writer, create_io_object


class MinhashPaths:
    def __init__(self, base_data_dir, base_logging_dir):
        self.signatures_dir = os.path.join(base_data_dir, 'signatures')
        self.buckets_dir = os.path.join(base_data_dir, 'buckets')
        self.remove_ids_dir = os.path.join(base_data_dir, 'remove_ids')
        self.removed_dir = os.path.join(base_data_dir, 'removed')
        
        if base_logging_dir is not None:
            self.logging_signatures_dir = os.path.join(base_logging_dir, 'signatures')
            self.logging_buckets_dir = os.path.join(base_logging_dir, 'buckets')
            self.logging_clusters_dir = os.path.join(base_logging_dir, 'clusters')
            self.logging_filter_dir = os.path.join(base_logging_dir, 'filter')
        else:
            self.logging_signatures_dir = None
            self.logging_buckets_dir = None
            self.logging_clusters_dir = None
            self.logging_filter_dir = None


def minhash_deduplication(
    INPUT_READER,
    OUTPUT_WRITER,
    EXCLUSION_WRITER,
    paths: MinhashPaths,
    hash_config: dict,
    minhash_config: dict,
    language: str,
    total_tasks: int,
    workers: int
):
    hash_config_obj = HashConfig(**hash_config)
    
    minhash_config_obj = MinhashConfig(
        hash_config=hash_config_obj,
        **minhash_config
    )

    stage1 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            MinhashDedupSignature(
                output_folder=paths.signatures_dir,
                config=minhash_config_obj,
                language=language
            ),
        ],
        logging_dir=paths.logging_signatures_dir,
        tasks=total_tasks,
        workers=workers
    )

    stage2 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=paths.signatures_dir,
                output_folder=paths.buckets_dir,
                config=minhash_config_obj,
            ),
        ],
        depends=stage1,
        logging_dir=paths.logging_buckets_dir,
        tasks=minhash_config_obj.num_buckets,
        workers=workers
    )

    stage3 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=paths.buckets_dir,
                output_folder=paths.remove_ids_dir,
                config=minhash_config_obj,
            ),
        ],
        depends=stage2,
        logging_dir=paths.logging_clusters_dir,
        tasks=1,
        workers=workers
    )

    stage4 = LocalPipelineExecutor(
        pipeline=[
            INPUT_READER,
            TokensCounter(),  
            MinhashDedupFilter(
                input_folder=paths.remove_ids_dir,
                exclusion_writer=EXCLUSION_WRITER,
            ),
            OUTPUT_WRITER
        ],
        depends=stage3,
        logging_dir=paths.logging_filter_dir,
        tasks=total_tasks,
        workers=workers
    )

    stage4.run()


def run_minhash_deduplication(path_to_yaml_config: str) -> None:
    with open(path_to_yaml_config, "r", encoding="utf-8") as f:
        yaml_config = yaml.safe_load(f) or {}

    if not yaml_config:
        raise ValueError("Config is empty or incorrect.")
    
    INPUT_READER, OUTPUT_WRITER = create_reader_and_writer(yaml_config)
        
    EXCLUSION_WRITER = create_io_object(
        cfg=yaml_config['exclusion_writer'],
        type='writer',
        use_adapter=False
    ) if yaml_config.get('exclusion_writer') else None
    
    paths = MinhashPaths(
        base_data_dir=yaml_config['base_data_dir'],
        base_logging_dir=yaml_config.get('base_logging_dir')
    )
    
    minhash_deduplication(
        INPUT_READER=INPUT_READER,
        OUTPUT_WRITER=OUTPUT_WRITER,
        EXCLUSION_WRITER=EXCLUSION_WRITER,
        paths=paths,
        hash_config=yaml_config['hash_config'],
        minhash_config=yaml_config['minhash_config'],
        language=yaml_config['language'],
        total_tasks=yaml_config['num_shards'],
        workers=yaml_config['workers']
    )