"""Regenerate training-history figures after smooth-validation-curves changes."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

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
    fit_model_for_history,
    plot_training_history,
    split_windowed_data_by_engine,
)

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
TUNER_DIR = PROJECT_ROOT / "data" / "processed" / "keras_tuner"
FIGURES_DIR = PROJECT_ROOT / "data" / "processed" / "figures"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RUNS = (
    ("lstm", "hyper_lstm_hyperband", "LSTM", "lstm_history_fit.weights.h5", "lstm_training_history.png"),
    ("bilstm", "hyper_bi_hyperband", "BiLSTM", "bilstm_history_fit.weights.h5", "bilstm_training_history.png"),
    (
        "transformer",
        "hyper_transformer_hyperband",
        "Transformer",
        "transformer_history_fit.weights.h5",
        "transformer_training_history.png",
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

    for kind, project, label, ckpt_name, fig_name in RUNS:
        print(f"=== {label}: loading best Hyperband model ===", flush=True)
        best_model = load_best_model(kind, x1, project)
        ckpt = PROCESSED_DIR / ckpt_name
        print(f"=== {label}: fit_model_for_history (up to 80 epochs) ===", flush=True)
        history = fit_model_for_history(
            best_model,
            x1,
            y1,
            x2,
            y2,
            ckpt,
        )
        plot_training_history(
            history,
            label,
            save_path=FIGURES_DIR / fig_name,
        )
        print(f"=== {label}: saved {FIGURES_DIR / fig_name} ===", flush=True)


if __name__ == "__main__":
    main()
