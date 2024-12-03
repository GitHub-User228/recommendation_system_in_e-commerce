import pendulum
from airflow import DAG
from airflow.utils.task_group import TaskGroup
from airflow.operators.python import PythonOperator
from airflow.operators.dummy_operator import DummyOperator

from scripts.utils import create_new_experiment
from scripts.components.preprocessing import PreprocessingComponent
from scripts.components.eda import EDAComponent
from scripts.components.matrix_builder import MatrixBuilderComponent
from scripts.components.features_generator import FeaturesGeneratorComponent
from scripts.components.als import ALSModelComponent
from scripts.components.bpr import BPRModelComponent
from scripts.components.item2item import Item2ItemModelComponent
from scripts.components.top_items import TopItemsModelComponent
from scripts.components.ensemble import EnsembleModelComponent
from scripts.settings import get_airflow_config, config
from scripts.messages import (
    send_telegram_success_message,
    send_telegram_failure_message,
)


LOG = True
IS_AIRFLOW = True

airflow_config = get_airflow_config()

with DAG(
    airflow_config.dag_id,
    start_date=pendulum.from_format(
        airflow_config.start_date, "YYYY-MM-DD", tz="UTC"
    ),
    schedule_interval=airflow_config.schedule_interval,
    on_success_callback=send_telegram_success_message,
    on_failure_callback=send_telegram_failure_message,
) as master_dag:

    if LOG:
        start = PythonOperator(
            task_id="create_new_experiment",
            python_callable=create_new_experiment,
            op_kwargs={
                "experiment_name_prefix": config["experiment_name_prefix"],
                "is_airflow": IS_AIRFLOW,
            },
        )
    else:
        start = DummyOperator(task_id="create_new_experiment")

    # Stage 1: Testing Recsys system and obtaining the ensemble model
    with TaskGroup(
        "recsys_testing_pipeline_stage", tooltip="Stage 1 Tasks"
    ) as stage_1:

        is_testing = True

        step1a = PythonOperator(
            task_id="preprocessing",
            python_callable=PreprocessingComponent(
                is_airflow=IS_AIRFLOW
            ).preprocess,
            op_kwargs={
                "log": LOG,
            },
        )

        step2a = PythonOperator(
            task_id="eda",
            python_callable=EDAComponent(
                show_graphs=False, is_airflow=IS_AIRFLOW
            ).analyze,
            op_kwargs={
                "log": LOG,
            },
        )

        step3a = PythonOperator(
            task_id="matrix_building",
            python_callable=MatrixBuilderComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).build,
            op_kwargs={
                "log": LOG,
            },
        )

        step4a = PythonOperator(
            task_id="feature_generation",
            python_callable=FeaturesGeneratorComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).generate,
            op_kwargs={
                "log": LOG,
            },
        )

        step51a = PythonOperator(
            task_id="base_model_als__fit",
            python_callable=ALSModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).fit,
        )
        step52a = PythonOperator(
            task_id="base_model_als__recommend",
            python_callable=ALSModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).recommend,
        )
        step53a = PythonOperator(
            task_id="base_model_als__evaluate",
            python_callable=ALSModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).evaluate,
        )

        step61a = PythonOperator(
            task_id="base_model_bpr__fit",
            python_callable=BPRModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).fit,
        )
        step62a = PythonOperator(
            task_id="base_model_bpr__recommend",
            python_callable=BPRModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).recommend,
        )
        step63a = PythonOperator(
            task_id="base_model_bpr__evaluate",
            python_callable=BPRModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).evaluate,
        )

        step71a = PythonOperator(
            task_id="base_model_item2item__fit",
            python_callable=Item2ItemModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).fit,
        )
        step72a = PythonOperator(
            task_id="base_model_item2item__recommend",
            python_callable=Item2ItemModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).recommend,
        )
        step73a = PythonOperator(
            task_id="base_model_item2item__evaluate",
            python_callable=Item2ItemModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).evaluate,
        )

        step81a = PythonOperator(
            task_id="base_model_top_items__fit",
            python_callable=TopItemsModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).fit,
        )
        step82a = PythonOperator(
            task_id="base_model_top_items__recommend",
            python_callable=TopItemsModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).recommend,
        )
        step83a = PythonOperator(
            task_id="base_model_top_items__evaluate",
            python_callable=TopItemsModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).evaluate,
        )

        step91a = PythonOperator(
            task_id="ensemble_model__prepare_data",
            python_callable=EnsembleModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).prepare_data,
        )
        step92a = PythonOperator(
            task_id="ensemble_model__fit",
            python_callable=EnsembleModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).fit,
        )

        step1a >> step2a >> step3a >> step4a
        if LOG:

            step54a = PythonOperator(
                task_id="base_model_als__log",
                python_callable=ALSModelComponent(
                    is_testing=is_testing, is_airflow=IS_AIRFLOW
                ).log,
            )
            step64a = PythonOperator(
                task_id="base_model_bpr__log",
                python_callable=BPRModelComponent(
                    is_testing=is_testing, is_airflow=IS_AIRFLOW
                ).log,
            )
            step74a = PythonOperator(
                task_id="base_model_item2item__log",
                python_callable=Item2ItemModelComponent(
                    is_testing=is_testing, is_airflow=IS_AIRFLOW
                ).log,
            )
            step84a = PythonOperator(
                task_id="base_model_top_items__log",
                python_callable=TopItemsModelComponent(
                    is_testing=is_testing, is_airflow=IS_AIRFLOW
                ).log,
            )
            step93a = PythonOperator(
                task_id="ensemble_model__log",
                python_callable=EnsembleModelComponent(
                    is_testing=is_testing, is_airflow=IS_AIRFLOW
                ).log,
            )

            step4a >> step51a >> step52a >> step53a >> step54a
            step4a >> step61a >> step62a >> step63a >> step64a
            step4a >> step71a >> step72a >> step73a >> step74a
            step4a >> step81a >> step82a >> step83a >> step84a
            (
                [step54a, step64a, step74a, step84a]
                >> step91a
                >> step92a
                >> step93a
            )
        else:
            step4a >> step51a >> step52a >> step53a
            step4a >> step61a >> step62a >> step63a
            step4a >> step71a >> step72a >> step73a
            step4a >> step81a >> step82a >> step83a
            [step53a, step63a, step73a, step83a] >> step91a >> step92a

    # Stage 2: Obtaining recommendations based on all the data and
    # trained ensemble model
    with TaskGroup(
        "recsys_pipeline_stage", tooltip="Stage 2 Tasks"
    ) as stage_2:

        is_testing = False

        step3b = PythonOperator(
            task_id="matrix_building",
            python_callable=MatrixBuilderComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).build,
            op_kwargs={
                "log": LOG,
            },
        )

        step4b = PythonOperator(
            task_id="feature_generation",
            python_callable=FeaturesGeneratorComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).generate,
            op_kwargs={
                "log": LOG,
            },
        )

        step51b = PythonOperator(
            task_id="base_model_als__fit",
            python_callable=ALSModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).fit,
        )
        step52b = PythonOperator(
            task_id="base_model_als__recommend",
            python_callable=ALSModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).recommend,
        )

        step61b = PythonOperator(
            task_id="base_model_bpr__fit",
            python_callable=BPRModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).fit,
        )
        step62b = PythonOperator(
            task_id="base_model_bpr__recommend",
            python_callable=BPRModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).recommend,
        )

        step71b = PythonOperator(
            task_id="base_model_item2item__fit",
            python_callable=Item2ItemModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).fit,
        )
        step72b = PythonOperator(
            task_id="base_model_item2item__recommend",
            python_callable=Item2ItemModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).recommend,
        )

        step81b = PythonOperator(
            task_id="base_model_top_items__fit",
            python_callable=TopItemsModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).fit,
        )
        step82b = PythonOperator(
            task_id="base_model_top_items__recommend",
            python_callable=TopItemsModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).recommend,
        )

        step91b = PythonOperator(
            task_id="ensemble_model__recommend",
            python_callable=EnsembleModelComponent(
                is_testing=is_testing, is_airflow=IS_AIRFLOW
            ).recommend,
        )

        step3b >> step4b
        if LOG:
            step53b = PythonOperator(
                task_id="base_model_als__log",
                python_callable=ALSModelComponent(
                    is_testing=is_testing, is_airflow=IS_AIRFLOW
                ).log,
            )
            step63b = PythonOperator(
                task_id="base_model_bpr__log",
                python_callable=BPRModelComponent(
                    is_testing=is_testing, is_airflow=IS_AIRFLOW
                ).log,
            )
            step73b = PythonOperator(
                task_id="base_model_item2item__log",
                python_callable=Item2ItemModelComponent(
                    is_testing=is_testing, is_airflow=IS_AIRFLOW
                ).log,
            )
            step83b = PythonOperator(
                task_id="base_model_top_items__log",
                python_callable=TopItemsModelComponent(
                    is_testing=is_testing, is_airflow=IS_AIRFLOW
                ).log,
            )
            step92b = PythonOperator(
                task_id="ensemble_model__log",
                python_callable=EnsembleModelComponent(
                    is_testing=is_testing, is_airflow=IS_AIRFLOW
                ).log,
            )
            step4b >> step51b >> step52b >> step53b
            step4b >> step61b >> step62b >> step63b
            step4b >> step71b >> step72b >> step73b
            step4b >> step81b >> step82b >> step83b
            [step53b, step63b, step73b, step83b] >> step91b >> step92b
        else:
            step4b >> step51b >> step52b
            step4b >> step61b >> step62b
            step4b >> step71b >> step72b
            step4b >> step81b >> step82b
            [step52b, step62b, step72b, step82b] >> step91b

    end = DummyOperator(task_id="end")

    start >> stage_1 >> stage_2 >> end
