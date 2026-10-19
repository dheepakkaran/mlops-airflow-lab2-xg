"""
Structural tests for the two DAGs.

These exist to pin the orchestration decisions in place: if someone puts the
TriggerDagRunOperator back, drops the asset outlet, or unmaps the training task,
a test fails rather than the pipeline quietly going back to the lab's behaviour.
"""

EXPECTED_TASKS = {
    "task_using_linked_owner",
    "load_data_task",
    "data_preprocessing_task",
    "separate_data_outputs_task",
    "build_save_model_task",
    "select_best_model_task",
}


def test_no_import_errors(dagbag):
    assert dagbag.import_errors == {}, f"DAG files failed to import: {dagbag.import_errors}"


def test_both_dags_are_loaded(dagbag):
    assert set(dagbag.dag_ids) == {"Airflow_Lab2", "Airflow_Lab2_Flask"}


def test_training_dag_task_ids(dagbag):
    dag = dagbag.dags["Airflow_Lab2"]
    assert set(dag.task_dict) == EXPECTED_TASKS


def test_trigger_operator_is_gone(dagbag):
    """The lab pushed the serving DAG with a TriggerDagRunOperator. Assets replaced it."""
    for dag_id in dagbag.dag_ids:
        for task in dagbag.dags[dag_id].tasks:
            assert "TriggerDagRun" not in type(task).__name__, (
                f"{dag_id}.{task.task_id} still pushes a downstream DAG"
            )


def test_model_is_published_as_an_asset_after_evaluation(dagbag):
    """The outlet belongs on the reduce task, so nothing is published untested."""
    dag = dagbag.dags["Airflow_Lab2"]
    outlets = {a.name for a in dag.task_dict["select_best_model_task"].outlets}
    assert "xg_model" in outlets

    producer = dag.task_dict["build_save_model_task"]
    assert not getattr(producer, "outlets", []), (
        "build_save_model_task must not publish; its models are candidates, not the result"
    )


def test_training_task_is_mapped(dagbag):
    """One training task instance per entry in TRAINING_SETS."""
    from src.model_development import TRAINING_SETS

    task = dagbag.dags["Airflow_Lab2"].task_dict["build_save_model_task"]
    assert getattr(task, "is_mapped", False), "build_save_model_task is no longer a mapped task"
    assert len(TRAINING_SETS) == 3


def test_serving_dag_is_asset_scheduled(dagbag):
    """It has no clock: it runs when a model is published, and never on a timer."""
    dag = dagbag.dags["Airflow_Lab2_Flask"]
    timetable = type(dag.timetable).__name__
    assert timetable == "AssetTriggeredTimetable", f"expected asset scheduling, got {timetable}"

    assets = {a.name for a in dag.timetable.asset_condition.objects}
    assert assets == {"xg_model"}


def test_serving_dag_cannot_run_concurrently(dagbag):
    """Its task binds a port, so a second simultaneous run could only fail."""
    assert dagbag.dags["Airflow_Lab2_Flask"].max_active_runs == 1
