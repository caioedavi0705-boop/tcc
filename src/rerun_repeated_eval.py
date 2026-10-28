"""Re-run 10× continued training + test eval for LSTM, BiLSTM, and Transformer."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import keras_tuner  # noqa: E402

from src.data import (  # noqa: E402
    assign_fd001_column_names,
    compute_test_rul,
    compute_train_rul,
    drop_missing_rows,
    drop_unused_sensor_columns,
    load_fd001_raw,
)
from src.features import create_fd001_windowed_features  # noqa: E402
from src.models import (  # noqa: E402
    _build_tuner_factory,
    run_repeated_train_eval,
    save_experiment_results,
    split_windowed_data_by_engine,
)

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
TUNER_DIR = PROJECT_ROOT / "data" / "processed" / "keras_tuner"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROCESSED_DIR / "results"
N_ITER = 10

RUNS = (
    ("lstm", "hyper_lstm_hyperband", "lstm_model.h5", "lstm_results.csv"),
    ("bilstm", "hyper_bi_hyperband", "bi_model.h5", "bilstm_results.csv"),
    (
        "transformer",
        "hyper_transformer_hyperband",
        "transformer_model.h5",
        "transformer_results.csv",
    ),
)


def load_best_model(model_kind: str, x_train, project_name: str):
    input_shape = (x_train.shape[1], x_train.shape[2])
    build_model = _build_tuner_factory(model_kind, input_shape)  # type: ignore[arg-type]
    tuner = keras_tuner.Hyperband(
        build_model,
        objective="val_loss",
        max_epochs=15,
        factor=3,
        hyperband_iterations=1,
        directory=str(TUNER_DIR),
        project_name=project_name,
        overwrite=False,
    )
    return tuner.get_best_models(num_models=1)[0]


def main() -> None:
    raw = load_fd001_raw(DATA_RAW_DIR)
    train_df, test_df = drop_missing_rows(raw.train, raw.test)
    train_df, test_df, rul_df = assign_fd001_column_names(train_df, test_df, raw.rul)
    train_df, test_df = drop_unused_sensor_columns(train_df, test_df)
    rul_train = compute_train_rul(train_df)
    rul_test = compute_test_rul(test_df, rul_df)

    windowed = create_fd001_windowed_features(
        train_df, test_df, rul_train, rul_test, window_size=30, step=1
    )
    x1, x2, y1, y2 = split_windowed_data_by_engine(
        windowed.x_train,
        windowed.y_train,
        windowed.train_window_engine_ids,
        test_size=0.2,
        random_state=1,
    )
    x_test, y_test = windowed.x_test, windowed.y_test

    for kind, project, ckpt_name, results_name in RUNS:
        label = kind.upper() if kind != "bilstm" else "BiLSTM"
        print(f"=== {label}: loading Hyperband best model ===", flush=True)
        best_model = load_best_model(kind, x1, project)
        checkpoint = PROCESSED_DIR / ckpt_name
        results_path = RESULTS_DIR / results_name
        print(f"=== {label}: run_repeated_train_eval ({N_ITER} iterations) ===", flush=True)
        resultados, _ = run_repeated_train_eval(
            best_model,
            x1,
            y1,
            x2,
            y2,
            x_test,
            y_test,
            checkpoint,
            n_iterations=N_ITER,
            capture_first_history=False,
        )
        save_experiment_results(resultados, results_path)
        print(resultados.to_string(index=False), flush=True)
        print(f"=== {label}: saved {results_path} ===", flush=True)

    print("=== Running Wilcoxon comparison (comparacao_wilcoxon.ipynb logic) ===", flush=True)
    from src.run_wilcoxon_comparison import main as run_wilcoxon  # noqa: WPS433

    run_wilcoxon(write_markdown=True)


if __name__ == "__main__":
    main()
