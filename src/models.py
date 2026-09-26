"""Treinamento, busca de hiperparâmetros e avaliação de modelos RUL."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import keras_tuner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Bidirectional, Dense, Dropout, LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ModelKind = Literal["lstm", "bilstm"]

LSTM_UNIT_CHOICES = [32, 64, 128, 256, 512]
DROPOUT_CHOICES = list(np.arange(0.2, 0.6, 0.1))
LEARNING_RATE_CHOICES = [0.01, 0.001, 0.0001]


def split_windowed_data(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide janelas temporais em treino e validação interna.

    Args:
        x: Tensor de features ``(n_amostras, window_size, n_sensores)``.
        y: Vetor de RUL alvo.
        test_size: Fração reservada para validação.
        random_state: Semente para reprodutibilidade.

    Returns:
        Tupla ``(x_train, x_val, y_train, y_val)`` em ``float32``.
    """
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    return (
        x_train.astype("float32"),
        x_val.astype("float32"),
        y_train.astype("float32"),
        y_val.astype("float32"),
    )


def build_model_lstm(
    hp: keras_tuner.HyperParameters,
    input_shape: tuple[int, int],
) -> Sequential:
    """Constrói arquitetura LSTM parametrizada para o Keras Tuner.

    Args:
        hp: Hiperparâmetros sugeridos pelo tuner.
        input_shape: ``(window_size, n_sensores)`` da primeira camada LSTM.

    Returns:
        Modelo Keras compilado com loss MAE e métrica MSE.
    """
    model = Sequential()
    n_layers_lstm = hp.Int("layers_lstm", 1, 4)
    n_layers_dense = hp.Int("layers_dense", 0, 1)
    for i in range(n_layers_lstm):
        return_seq = i < n_layers_lstm - 1
        if i == 0:
            model.add(
                LSTM(
                    hp.Choice(f"lstm_{i}", LSTM_UNIT_CHOICES),
                    activation="tanh",
                    return_sequences=True,
                    input_shape=input_shape,
                )
            )
        else:
            model.add(
                LSTM(
                    hp.Choice(f"lstm_{i}", LSTM_UNIT_CHOICES),
                    activation="tanh",
                    return_sequences=return_seq,
                )
            )
        model.add(Dropout(hp.Choice(f"dropout_{i}", DROPOUT_CHOICES)))
    for i in range(n_layers_dense):
        model.add(
            Dense(
                hp.Choice(f"dense_{i}", LSTM_UNIT_CHOICES),
                activation="relu",
            )
        )
    model.add(Dense(1, activation="linear"))
    model.compile(
        optimizer=RMSprop(
            learning_rate=hp.Choice("learning_rate", LEARNING_RATE_CHOICES)
        ),
        loss="mae",
        metrics=["mean_squared_error"],
    )
    return model


def build_model_bilstm(
    hp: keras_tuner.HyperParameters,
    input_shape: tuple[int, int],
) -> Sequential:
    """Constrói arquitetura BiLSTM parametrizada para o Keras Tuner.

    Args:
        hp: Hiperparâmetros sugeridos pelo tuner.
        input_shape: ``(window_size, n_sensores)`` da primeira camada.

    Returns:
        Modelo Keras compilado com loss MAE e métrica MSE.
    """
    model = Sequential()
    n_layers_bi = hp.Int("layers_bi", 1, 4)
    n_layers_dense = hp.Int("layers_dense", 0, 1)
    for i in range(n_layers_bi):
        return_seq = i < n_layers_bi - 1
        if i == 0:
            model.add(
                Bidirectional(
                    LSTM(
                        hp.Choice(f"bi_{i}", LSTM_UNIT_CHOICES),
                        activation="tanh",
                        return_sequences=True,
                    ),
                    input_shape=input_shape,
                )
            )
        else:
            model.add(
                Bidirectional(
                    LSTM(
                        hp.Choice(f"bi_{i}", LSTM_UNIT_CHOICES),
                        activation="tanh",
                        return_sequences=return_seq,
                    )
                )
            )
        model.add(Dropout(hp.Choice(f"dropout_{i}", DROPOUT_CHOICES)))
    for i in range(n_layers_dense):
        model.add(
            Dense(
                hp.Choice(f"dense_{i}", LSTM_UNIT_CHOICES),
                activation="relu",
            )
        )
    model.add(Dense(1, activation="linear"))
    model.compile(
        optimizer=RMSprop(
            learning_rate=hp.Choice("learning_rate", LEARNING_RATE_CHOICES)
        ),
        loss="mae",
        metrics=["mean_squared_error"],
    )
    return model


def _build_tuner_factory(
    model_kind: ModelKind,
    input_shape: tuple[int, int],
):
    """Retorna função ``build_model(hp)`` fechada sobre ``input_shape``."""

    def build_model(hp: keras_tuner.HyperParameters) -> Sequential:
        if model_kind == "lstm":
            return build_model_lstm(hp, input_shape)
        return build_model_bilstm(hp, input_shape)

    return build_model


def summarize_lstm_hyperparameters(param_values: dict[str, Any]) -> dict[str, Any]:
    """Organiza hiperparâmetros ótimos do LSTM como no notebook.

    Args:
        param_values: Dicionário ``values`` do trial vencedor do Keras Tuner.

    Returns:
        Mapa legível de hiperparâmetros (inclui camada de saída com valor 1).
    """
    n_lstm = int(param_values.get("layers_lstm", 1))
    n_dense = int(param_values.get("layers_dense", 1)) + 1
    best_params: dict[str, Any] = {}
    for k in range(n_lstm):
        best_params[f"lstm_{k}"] = param_values[f"lstm_{k}"]
        best_params[f"dropout_{k}"] = param_values[f"dropout_{k}"]
    for k in range(n_dense):
        key = f"dense_{k}"
        if k < n_dense - 1:
            best_params[key] = param_values[key]
        else:
            best_params[key] = 1
    best_params["learning_rate"] = param_values["learning_rate"]
    return best_params


def summarize_bilstm_hyperparameters(param_values: dict[str, Any]) -> dict[str, Any]:
    """Organiza hiperparâmetros ótimos do BiLSTM como no notebook.

    Args:
        param_values: Dicionário ``values`` do trial vencedor do Keras Tuner.

    Returns:
        Mapa legível de hiperparâmetros (inclui camada de saída com valor 1).
    """
    n_bi = int(param_values.get("layers_bi", 1))
    n_dense = int(param_values.get("layers_dense", 1)) + 1
    best_params: dict[str, Any] = {}
    for k in range(n_bi):
        best_params[f"bi_{k}"] = param_values[f"bi_{k}"]
        best_params[f"dropout_{k}"] = param_values[f"dropout_{k}"]
    for k in range(n_dense):
        key = f"dense_{k}"
        if k < n_dense - 1:
            best_params[key] = param_values[key]
        else:
            best_params[key] = 1
    best_params["learning_rate"] = param_values["learning_rate"]
    return best_params


def search_hyperparameters(
    model_kind: ModelKind,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    tuner_directory: str | Path,
    project_name: str,
    max_trials: int = 15,
    search_epochs: int = 5,
    overwrite: bool = True,
) -> tuple[Sequential, dict[str, Any]]:
    """Executa busca bayesiana de hiperparâmetros.

    Args:
        model_kind: ``\"lstm\"`` ou ``\"bilstm\"``.
        x_train: Features de treino.
        y_train: Alvos de treino.
        x_val: Features de validação.
        y_val: Alvos de validação.
        tuner_directory: Diretório base para logs do Keras Tuner.
        project_name: Nome do subprojeto (ex.: ``hyper_lstm``).
        max_trials: Número máximo de trials.
        search_epochs: Épocas por trial na fase de busca.
        overwrite: Se ``True``, sobrescreve resultados anteriores do projeto.

    Returns:
        Tupla ``(best_model, best_params_summary)``.
    """
    input_shape = (x_train.shape[1], x_train.shape[2])
    build_model = _build_tuner_factory(model_kind, input_shape)
    tuner = keras_tuner.BayesianOptimization(
        build_model,
        objective="val_loss",
        max_trials=max_trials,
        directory=str(tuner_directory),
        project_name=project_name,
        overwrite=overwrite,
    )
    tuner.search(
        x_train,
        y_train,
        epochs=search_epochs,
        validation_data=(x_val, y_val),
    )
    best_model = tuner.get_best_models(num_models=1)[0]
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    param_values = best_trial.hyperparameters.get_config()["values"]
    if model_kind == "lstm":
        summary = summarize_lstm_hyperparameters(param_values)
    else:
        summary = summarize_bilstm_hyperparameters(param_values)
    return best_model, summary


def create_training_callbacks(
    checkpoint_path: str | Path,
    monitor: str = "val_loss",
    patience: int = 5,
) -> list[EarlyStopping | ModelCheckpoint]:
    """Cria callbacks de early stopping e checkpoint.

    Args:
        checkpoint_path: Caminho do arquivo ``.h5`` para salvar melhor peso.
        monitor: Métrica monitorada.
        patience: Épocas sem melhora antes de parar.

    Returns:
        Lista de callbacks para ``model.fit``.
    """
    return [
        EarlyStopping(monitor=monitor, patience=patience),
        ModelCheckpoint(str(checkpoint_path), monitor=monitor),
    ]


def compute_rul_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calcula métricas de regressão para previsão de RUL.

    Args:
        y_true: RUL verdadeiro.
        y_pred: RUL previsto.

    Returns:
        Dicionário com chaves ``MSE``, ``RMSE``, ``MAE`` e ``R2``.
    """
    y_pred_flat = np.asarray(y_pred, dtype=float).reshape(-1)
    y_true_flat = np.asarray(y_true, dtype=float).reshape(-1)
    mse = float(mean_squared_error(y_true_flat, y_pred_flat))
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true_flat, y_pred_flat)),
        "R2": float(r2_score(y_true_flat, y_pred_flat)),
    }


def predict_rul(model: Sequential, x: np.ndarray) -> np.ndarray:
    """Gera previsões de RUL achatadas.

    Args:
        model: Modelo treinado.
        x: Features de entrada.

    Returns:
        Vetor 1D de previsões ``float``.
    """
    predictions = model.predict(x, verbose=0).reshape(-1)
    return np.array([float(value) for value in predictions])


def plot_training_history(history: Any, model_label: str) -> None:
    """Plota loss (MAE) e MSE de treino e validação (primeira iteração)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    ax1.plot(history.history["loss"], label=f"Loss do treino ({model_label})")
    ax1.plot(history.history["val_loss"], label=f"Loss da validação ({model_label})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MAE)")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2.plot(
        history.history["mean_squared_error"],
        label=f"MSE do treino ({model_label})",
    )
    ax2.plot(
        history.history["val_mean_squared_error"],
        label=f"MSE da validação ({model_label})",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Erro Quadrático Médio (MSE)")
    ax2.legend(loc="upper right")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(f"Avaliação do treinamento ({model_label})")
    fig.tight_layout()
    plt.show()


def plot_rul_prediction_diagnostics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    *,
    model_label: str,
    rul_limit: int = 130,
    n_samples: int = 180,
    random_state: int | None = None,
) -> None:
    """Plota amostra ordenada e scatter RUL real vs previsto."""
    y_pred_flat = np.asarray(y_pred, dtype=float).reshape(-1)
    y_true_flat = np.asarray(y_test, dtype=float).reshape(-1)
    rng = np.random.default_rng(random_state)
    n = min(n_samples, len(y_pred_flat))
    indices = rng.choice(len(y_pred_flat), size=n, replace=False)
    y_real_sample = y_true_flat[indices]
    y_pred_sample = y_pred_flat[indices]
    order = np.argsort(y_real_sample)
    y_real_sample = y_real_sample[order]
    y_pred_sample = y_pred_sample[order]

    plt.figure(figsize=(10, 6))
    plt.scatter(range(n), y_pred_sample, c="blue", label="RUL previsto")
    plt.scatter(range(n), y_real_sample, c="red", label="RUL real")
    plt.title(f"RUL real vs RUL previsto ({n} amostras) ({model_label})")
    plt.ylabel("RUL")
    plt.xlabel("Index")
    plt.legend(loc="lower right")
    for i in range(n):
        plt.plot(
            [i, i],
            [y_pred_sample[i], y_real_sample[i]],
            ls="--",
            c="black",
            alpha=0.7,
        )
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true_flat, y_pred_flat, c="blue")
    plt.plot([0, rul_limit], [0, rul_limit], ls="--", c="red", alpha=0.8)
    plt.title(f"RUL real vs RUL previsto ({model_label})")
    plt.ylabel("RUL previsto")
    plt.xlabel("RUL real")
    plt.show()


def run_repeated_train_eval(
    model: Sequential,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    checkpoint_path: str | Path,
    n_iterations: int = 10,
    epochs: int = 30,
    batch_size: int = 200,
    patience: int = 5,
    capture_first_history: bool = False,
) -> tuple[pd.DataFrame, Any | None]:
    """Repete treino + avaliação no conjunto de teste (como no notebook).

    O mesmo objeto ``model`` é reutilizado em todas as iterações.

    Args:
        model: Modelo com arquitetura já definida (ex.: melhor trial do tuner).
        x_train: Features de treino.
        y_train: Alvos de treino.
        x_val: Features de validação interna.
        y_val: Alvos de validação interna.
        x_test: Features de teste hold-out.
        y_test: RUL verdadeiro no teste.
        checkpoint_path: Arquivo para ``ModelCheckpoint``.
        n_iterations: Número de repetições experimentais.
        epochs: Épocas máximas por iteração.
        batch_size: Tamanho do batch.
        patience: Paciência do early stopping.

    Returns:
        Tupla ``(results, first_history)``. ``first_history`` é o objeto
        retornado por ``model.fit`` na primeira iteração quando
        ``capture_first_history=True``; caso contrário, ``None``.
    """
    callbacks = create_training_callbacks(checkpoint_path, patience=patience)
    rows: list[dict[str, Any]] = []
    first_history: Any | None = None
    for iteration in range(n_iterations):
        start = time.time()
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        if iteration == 0 and capture_first_history:
            first_history = history
        y_pred = predict_rul(model, x_test)
        elapsed = time.time() - start
        metrics = compute_rul_metrics(y_test, y_pred)
        rows.append(
            {
                "Iteração": iteration + 1,
                **metrics,
                "tempo": elapsed,
            }
        )
    return pd.DataFrame(rows), first_history


def summarize_experiment_results(results: pd.DataFrame) -> pd.DataFrame:
    """Resume média e desvio padrão das métricas por modelo.

    Args:
        results: Saída de ``run_repeated_train_eval`` (várias iterações).

    Returns:
        DataFrame indexado por métrica com colunas ``Média`` e ``Desvio padrão``.
    """
    numeric = results.drop(columns=["Iteração"], errors="ignore")
    return pd.DataFrame(
        {
            "Média": numeric.mean(),
            "Desvio padrão": numeric.std(),
        }
    )


def compare_models_wilcoxon(
    results_lstm: pd.DataFrame,
    results_bilstm: pd.DataFrame,
    alpha: float = 0.05,
    exclude_columns: tuple[str, ...] = ("Iteração", "tempo"),
) -> pd.DataFrame:
    """Compara LSTM e BiLSTM com teste de Wilcoxon pareado por iteração.

    Args:
        results_lstm: Resultados repetidos do LSTM.
        results_bilstm: Resultados repetidos do BiLSTM.
        alpha: Nível de significância.
        exclude_columns: Colunas ignoradas na comparação.

    Returns:
        DataFrame com ``metrica``, ``p_value`` e ``significativo``.
    """
    df_lstm = results_lstm.drop(columns=list(exclude_columns), errors="ignore")
    df_bilstm = results_bilstm.drop(columns=list(exclude_columns), errors="ignore")
    differences = df_lstm - df_bilstm
    rows: list[dict[str, Any]] = []
    for metric in differences.columns:
        p_value = float(wilcoxon(differences[metric]).pvalue)
        rows.append(
            {
                "metrica": metric,
                "p_value": p_value,
                "significativo": p_value < alpha,
            }
        )
    return pd.DataFrame(rows)
