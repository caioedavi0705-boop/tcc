"""Criação de features (escala e janelas temporais) para modelagem RUL."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data import ID_COLUMN

DEFAULT_WINDOW_SIZE = 30
DEFAULT_WINDOW_STEP = 1


class FD001WindowedFeatures(NamedTuple):
    """Tensores e DataFrames intermediários prontos para modelos sequenciais."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler
    features_train: pd.DataFrame
    features_test: pd.DataFrame


def sensor_feature_columns(features: pd.DataFrame) -> list[str]:
    """Retorna nomes das colunas de sensores (exclui ID e ciclo).

    Args:
        features: DataFrame com ID e ciclo nas duas primeiras colunas.

    Returns:
        Lista de nomes das colunas a partir do índice 2.
    """
    return features.columns[2:].tolist()


def scale_sensor_features(
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    scaler: MinMaxScaler | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Normaliza sensores com ``MinMaxScaler`` (fit no treino).

    Args:
        features_train: DataFrame de treino (ID, ciclo + sensores).
        features_test: DataFrame de teste.
        scaler: Scaler já ajustado; se ``None``, ajusta um novo no treino.

    Returns:
        Tupla ``(train_scaled, test_scaled, scaler)``.
    """
    train_out = features_train.copy()
    test_out = features_test.copy()
    feature_cols = sensor_feature_columns(train_out)
    fitted = scaler or MinMaxScaler()
    if scaler is None:
        train_out[feature_cols] = fitted.fit_transform(train_out[feature_cols])
    else:
        train_out[feature_cols] = fitted.transform(train_out[feature_cols])
    test_out[feature_cols] = fitted.transform(test_out[feature_cols])
    return train_out, test_out, fitted


def time_window(
    data: pd.DataFrame,
    rul: list[int],
    window_size: int,
    step: int,
    id_column: str = ID_COLUMN,
) -> tuple[np.ndarray, np.ndarray]:
    """Monta janelas deslizantes de sensores e o RUL no fim da janela.

    Args:
        data: Features escalonadas (sensores a partir da coluna índice 2).
        rul: Targets por linha, na mesma ordem de ``data``.
        window_size: Tamanho da janela temporal.
        step: Deslocamento entre janelas.
        id_column: Coluna de ID do motor.

    Returns:
        Tupla ``(x, y)`` com ``x`` de shape
        ``(n_amostras, window_size, n_sensores)``.
    """
    windows: list[np.ndarray] = []
    targets: list[int] = []
    offset = 0
    for motor_id in data[id_column].unique():
        engine = data[data[id_column] == motor_id]
        for start in range(0, len(engine) - window_size + 1, step):
            end = start + window_size
            windows.append(engine.iloc[start:end, 2:].values)
            targets.append(rul[offset + end - 1])
        offset += len(engine)
    return np.array(windows), np.array(targets)


def create_fd001_windowed_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    rul_train: list[int],
    rul_test: list[int],
    window_size: int = DEFAULT_WINDOW_SIZE,
    step: int = DEFAULT_WINDOW_STEP,
    scaler: MinMaxScaler | None = None,
) -> FD001WindowedFeatures:
    """Cria features escalonadas e tensores com janelas temporais.

    Espera ``train``/``test`` já limpos (``src.data``) e listas de RUL alinhadas.

    Args:
        train: Treino com sensores selecionados (pós-``drop_unused_sensor_columns``).
        test: Teste com sensores selecionados.
        rul_train: Target RUL por linha de treino.
        rul_test: Target RUL por linha de teste.
        window_size: Comprimento da janela (padrão 30).
        step: Passo da janela (padrão 1).
        scaler: Scaler opcional já ajustado.

    Returns:
        ``FD001WindowedFeatures`` com arrays, scaler e DataFrames escalonados.
    """
    features_train, features_test, fitted_scaler = scale_sensor_features(
        train, test, scaler=scaler
    )
    x_train, y_train = time_window(
        features_train, rul_train, window_size, step
    )
    x_test, y_test = time_window(features_test, rul_test, window_size, step)
    return FD001WindowedFeatures(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        scaler=fitted_scaler,
        features_train=features_train,
        features_test=features_test,
    )
