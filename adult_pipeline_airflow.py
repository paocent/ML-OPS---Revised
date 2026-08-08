"""
Airflow DAG entry point for the Adult Census Income TFX pipeline.

This file only exists for Airflow's DAG discovery (it scans every .py file
under its dags/ folder for a module-level DAG object) — the actual pipeline
is defined once in adult_pipeline_definition.py and shared with the
standalone local runner (adult_pipeline_local.py).

To run this via Airflow, copy/symlink this whole adult_census_tfx/ folder
into your $AIRFLOW_HOME/dags/ directory, then trigger the
"adult_census_tfx" DAG from the Airflow UI or `airflow dags trigger`.
"""
import datetime

from tfx.orchestration.airflow.airflow_dag_runner import (
    AirflowDagRunner,
    AirflowPipelineConfig,
)

import adult_pipeline_definition as pdef

_airflow_config = {
    "schedule_interval": None,
    "start_date": datetime.datetime(2026, 1, 1),
    "params": {
        "accuracy_threshold": pdef.ACCURACY_THRESHOLD,
        "data_root": pdef.DATA_ROOT,
    },
}

DAG = AirflowDagRunner(
    AirflowPipelineConfig(_airflow_config)
).run(
    pdef.create_pipeline(
        pipeline_name=pdef.PIPELINE_NAME,
        pipeline_root=pdef.PIPELINE_ROOT,
        data_root=pdef.DATA_ROOT,
        metadata_path=pdef.METADATA_PATH,
        serving_model_dir=pdef.SERVING_MODEL_DIR,
        train_num_steps=pdef.TRAIN_NUM_STEPS,
        eval_num_steps=pdef.EVAL_NUM_STEPS,
        accuracy_threshold=pdef.ACCURACY_THRESHOLD,
        use_runtime_data_root_param=True,
    )
)
