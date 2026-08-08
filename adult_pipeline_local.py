"""
Run the whole Adult Census TFX pipeline with a single command — no Airflow
setup required:

    python adult_pipeline_local.py

This uses TFX's LocalDagRunner, which executes every component
(ExampleGen -> ... -> Pusher) in-process, one after another, writing all
artifacts under PIPELINE_ROOT and recording them in the MLMD SQLite DB at
METADATA_PATH (see adult_pipeline_definition.py for those paths).

After the run finishes, this script also queries MLMD and prints the
things the assignment explicitly asks you to "display"/"print":
  - Step 1: the ExampleGen train/eval split URIs
  - Step 2: the StatisticsGen artifact URIs (open the .pb files with
            tfdv.load_statistics() / tfdv.visualize_statistics() in a
            notebook to see the actual histograms)
  - Step 4: ExampleValidator anomalies, and a healthy/unhealthy verdict

Use Airflow instead (adult_pipeline_airflow.py) when you need the
"Pipeline Execution Evidence" screenshots the rubric asks for (DAG view,
task logs, all-green run) — this script does the identical pipeline run
but skips the Airflow UI entirely, so it's for fast local iteration.
"""
import glob
import os

from tfx.orchestration.local.local_dag_runner import LocalDagRunner

import adult_pipeline_definition as pdef


def run_pipeline():
    pipeline_obj = pdef.create_pipeline(
        pipeline_name=pdef.PIPELINE_NAME,
        pipeline_root=pdef.PIPELINE_ROOT,
        data_root=pdef.DATA_ROOT,
        metadata_path=pdef.METADATA_PATH,
        serving_model_dir=pdef.SERVING_MODEL_DIR,
        train_num_steps=pdef.TRAIN_NUM_STEPS,
        eval_num_steps=pdef.EVAL_NUM_STEPS,
        accuracy_threshold=pdef.ACCURACY_THRESHOLD,
        # LocalDagRunner has no trigger-time parameter UI (unlike Airflow),
        # so data_root is passed as a plain string here.
        use_runtime_data_root_param=False,
    )
    LocalDagRunner().run(pipeline_obj)


def _print_header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def inspect_outputs():
    """Best-effort MLMD inspection for the Step 1/2/4 print requirements.

    Artifact directory layout differs a bit across TFX versions, so this
    walks each artifact's URI looking for the relevant files rather than
    hard-coding one exact path. If your installed TFX version lays things
    out differently, use this as a starting point and adjust the glob
    patterns — or just browse PIPELINE_ROOT directly.
    """
    from tfx.orchestration import metadata as tfx_metadata

    connection_config = tfx_metadata.sqlite_metadata_connection_config(
        pdef.METADATA_PATH
    )
    with tfx_metadata.Metadata(connection_config) as md:
        store = md.store

        _print_header("STEP 1 — ExampleGen split URIs")
        for artifact in store.get_artifacts_by_type("Examples"):
            print(f"  artifact id={artifact.id}  uri={artifact.uri}")
            for split_dir in sorted(glob.glob(os.path.join(artifact.uri, "Split-*"))):
                print(f"    split: {split_dir}")

        _print_header("STEP 2 — StatisticsGen artifact URIs")
        print("  (open these with tfdv.load_statistics() / "
              "tfdv.visualize_statistics() in a notebook)")
        for artifact in store.get_artifacts_by_type("ExampleStatistics"):
            print(f"  artifact id={artifact.id}  uri={artifact.uri}")

        _print_header("STEP 4 — ExampleValidator anomalies")
        try:
            import tensorflow_data_validation as tfdv
        except ImportError:
            print("  tensorflow_data_validation not importable in this "
                  "environment; open the anomalies artifact manually.")
            return

        any_anomalies = False
        anomaly_artifacts = store.get_artifacts_by_type("ExampleAnomalies")
        if not anomaly_artifacts:
            print("  No ExampleAnomalies artifacts found — did the "
                  "ExampleValidator component run?")
        for artifact in anomaly_artifacts:
            candidate_files = glob.glob(
                os.path.join(artifact.uri, "**", "*"), recursive=True
            )
            candidate_files = [f for f in candidate_files if os.path.isfile(f)]
            for f in candidate_files:
                try:
                    anomalies = tfdv.load_anomalies_text(f)
                except Exception:
                    continue
                print(f"  {f}")
                if len(anomalies.anomaly_info) == 0:
                    print("    no anomalies detected in this split")
                else:
                    any_anomalies = True
                    for feature_name, info in anomalies.anomaly_info.items():
                        print(f"    ANOMALY [{feature_name}]: "
                              f"{info.short_description}")

        _print_header("Data pipeline health check")
        if any_anomalies:
            print("  UNHEALTHY — anomalies were detected. Review the "
                  "ExampleValidator output above before trusting "
                  "downstream training/eval results.")
        else:
            print("  HEALTHY — no anomalies detected against the "
                  "inferred schema.")


if __name__ == "__main__":
    run_pipeline()
    inspect_outputs()
