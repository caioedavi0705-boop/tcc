"""Carregamento e limpeza dos dados NASA C-MAPSS FD001."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ID_COLUMN = "ID"
CYCLE_COLUMN = "Nº Ciclos"
RUL_COLUMN = "rul"
DEFAULT_RUL_LIMIT = 130

FD001_SENSOR_COLUMN_NAMES: list[str] = [
    ID_COLUMN,
    CYCLE_COLUMN,
    "Altitude [ft]",
    "Mach",
    "TRA",
    "T2 [°R]",
    "T24 [°R]",
    "T30 [°R]",
    "T50[°R]",
    "P2 [psia]",
    "P15 [psia]",
    "P30 [psia]",
    "Nf [rpm]",
    "Nc [rpm]",
    "epr [-]",
    "Ps30 [psia]",
    "phi [pps/psi]",
    "NRf [rpm]",
    "NRc [rpm]",
    "BPR [-]",
    "farB [-]",
    "htBleed [-]",
    "Nf_dmd [rpm]",
    "PCNfR_dmd [rpm]",
    "W31 [lbm/s]",
    "W32 [lbm/s]",
]

COLUMNS_TO_DROP: list[str] = [
    "TRA",
    "T2 [°R]",
    "P2 [psia]",
    "epr [-]",
    "farB [-]",
    "Nf_dmd [rpm]",
    "PCNfR_dmd [rpm]",
]


class FD001RawData(NamedTuple):
    """Conjuntos brutos do dataset FD001."""

    train: pd.DataFrame
    test: pd.DataFrame
    rul: pd.DataFrame


def load_fd001_raw(data_dir: str | Path) -> FD001RawData:
    """Carrega os arquivos de treino, teste e RUL do FD001.

    Args:
        data_dir: Diretório que contém ``train_FD001.txt``, ``test_FD001.txt`` e
            ``RUL_FD001.txt``.

    Returns:
        Tupla nomeada com os três DataFrames lidos (sem cabeçalho).
    """
    base = Path(data_dir)
    read_kwargs = {"sep": r"\s+", "header": None}
    train = pd.read_csv(base / "train_FD001.txt", **read_kwargs)
    test = pd.read_csv(base / "test_FD001.txt", **read_kwargs)
    rul = pd.read_csv(base / "RUL_FD001.txt", **read_kwargs)
    return FD001RawData(train=train, test=test, rul=rul)


def drop_missing_rows(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove linhas com valores ausentes nos conjuntos de treino e teste.

    Args:
        train: DataFrame de treino.
        test: DataFrame de teste.

    Returns:
        Par ``(train, test)`` sem linhas nulas.
    """
    return train.dropna().copy(), test.dropna().copy()


def assign_fd001_column_names(
    train: pd.DataFrame,
    test: pd.DataFrame,
    rul: pd.DataFrame,
    sensor_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Atribui nomes de colunas aos DataFrames do FD001.

    Args:
        train: DataFrame de treino.
        test: DataFrame de teste.
        rul: DataFrame com RUL verdadeiro por motor (teste).
        sensor_columns: Lista de nomes das colunas de sensores; usa
            ``FD001_SENSOR_COLUMN_NAMES`` se omitida.

    Returns:
        Tupla ``(train, test, rul)`` com colunas renomeadas.
    """
    columns = sensor_columns or FD001_SENSOR_COLUMN_NAMES
    train_out = train.copy()
    test_out = test.copy()
    rul_out = rul.copy()
    train_out.columns = columns
    test_out.columns = columns
    rul_out.columns = [RUL_COLUMN]
    return train_out, test_out, rul_out


def drop_unused_sensor_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns_to_drop: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove sensores não utilizados na modelagem.

    Args:
        train: DataFrame de treino com cabeçalho.
        test: DataFrame de teste com cabeçalho.
        columns_to_drop: Colunas a remover; usa ``COLUMNS_TO_DROP`` se omitida.

    Returns:
        Par ``(features_train, features_test)`` apenas com variáveis mantidas.
    """
    drop_cols = columns_to_drop or COLUMNS_TO_DROP
    return (
        train.drop(columns=drop_cols).copy(),
        test.drop(columns=drop_cols).copy(),
    )


def scale_sensor_features(
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    scaler: MinMaxScaler | None = None,
    id_column: str = ID_COLUMN,
    cycle_column: str = CYCLE_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Normaliza colunas de sensores com ``MinMaxScaler``.

    Ajusta o scaler apenas no treino e aplica a mesma transformação no teste.
    Colunas de identificação (ID e ciclo) não são escalonadas.

    Args:
        features_train: Features de treino.
        features_test: Features de teste.
        scaler: Scaler pré-ajustado; se ``None``, um novo scaler é criado e
            ajustado no treino.
        id_column: Nome da coluna de ID do motor.
        cycle_column: Nome da coluna de número de ciclos.

    Returns:
        Tupla ``(train_scaled, test_scaled, scaler)`` com sensores no intervalo
        [0, 1].
    """
    train_out = features_train.copy()
    test_out = features_test.copy()
    feature_cols = train_out.columns[2:].tolist()
    fitted = scaler or MinMaxScaler()
    if scaler is None:
        train_out[feature_cols] = fitted.fit_transform(train_out[feature_cols])
    else:
        train_out[feature_cols] = fitted.transform(train_out[feature_cols])
    test_out[feature_cols] = fitted.transform(test_out[feature_cols])
    return train_out, test_out, fitted


def compute_train_rul(
    train: pd.DataFrame,
    limit: int = DEFAULT_RUL_LIMIT,
    id_column: str = ID_COLUMN,
    cycle_column: str = CYCLE_COLUMN,
) -> list[int]:
    """Calcula o RUL por linha do conjunto de treino com truncamento.

    Para cada motor, o RUL é ``max_ciclos - ciclo_atual``, limitado a
    ``limit`` quando ainda restam mais de ``limit`` ciclos até a falha.

    Args:
        train: DataFrame de treino com colunas de ID e ciclo.
        limit: Valor máximo de RUL (piecewise linear, padrão NASA).
        id_column: Nome da coluna de ID do motor.
        cycle_column: Nome da coluna de número de ciclos.

    Returns:
        Lista de RUL (inteiros), uma entrada por linha de ``train`` na ordem
        original agrupada por motor.
    """
    max_cycles = train.groupby(id_column)[cycle_column].max().reset_index()
    rul_values: list[int] = []
    for motor_id in train[id_column].unique():
        engine = train[train[id_column] == motor_id]
        motor_max = max_cycles[cycle_column].iloc[int(motor_id) - 1]
        for cycle_index in range(len(engine[cycle_column])):
            if cycle_index + 1 <= motor_max - limit:
                rul_values.append(limit)
            else:
                rul_values.append(int(motor_max - engine[cycle_column].iloc[cycle_index]))
    return [int(value) for value in rul_values]


def compute_test_rul(
    test: pd.DataFrame,
    rul_remaining: pd.DataFrame,
    limit: int = DEFAULT_RUL_LIMIT,
    id_column: str = ID_COLUMN,
    cycle_column: str = CYCLE_COLUMN,
    rul_column: str = RUL_COLUMN,
) -> list[int]:
    """Calcula o RUL por linha do conjunto de teste.

    Combina ciclos restantes observados com o RUL verdadeiro ao final do
    teste (``rul_remaining``) e aplica o mesmo truncamento em ``limit``.

    Args:
        test: DataFrame de teste com colunas de ID e ciclo.
        rul_remaining: RUL verdadeiro por motor ao fim da janela de teste.
        limit: Valor máximo de RUL.
        id_column: Nome da coluna de ID do motor.
        cycle_column: Nome da coluna de número de ciclos.
        rul_column: Nome da coluna de RUL em ``rul_remaining``.

    Returns:
        Lista de RUL (inteiros), uma entrada por linha de ``test``.
    """
    max_cycles = test.groupby(id_column)[cycle_column].max().reset_index()
    rul_values: list[int] = []
    for motor_id in test[id_column].unique():
        engine = test[test[id_column] == motor_id]
        motor_max = max_cycles[cycle_column].iloc[int(motor_id) - 1]
        true_rul_at_end = rul_remaining[rul_column].iloc[int(motor_id) - 1]
        for cycle_index in range(len(engine[cycle_column])):
            partial_rul = motor_max - engine[cycle_column].iloc[cycle_index]
            value = partial_rul + true_rul_at_end
            if value >= limit:
                rul_values.append(limit)
            else:
                rul_values.append(int(value))
    return [int(value) for value in rul_values]


def time_window(
    data: pd.DataFrame,
    rul: list[int],
    window_size: int,
    step: int,
    id_column: str = ID_COLUMN,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera janelas temporais de sensores e o RUL alvo correspondente.

    Args:
        data: DataFrame com ID na primeira coluna e sensores a partir da
            terceira coluna (índice 2).
        rul: Lista de RUL alinhada à ordem das linhas em ``data`` (por motor).
        window_size: Comprimento da janela em ciclos.
        step: Passo entre janelas consecutivas do mesmo motor.
        id_column: Nome da coluna de ID do motor.

    Returns:
        Tupla ``(x, y)`` com arrays ``x`` de forma
        ``(n_amostras, window_size, n_sensores)`` e ``y`` unidimensional.
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


def prepare_fd001_windowed_data(
    data_dir: str | Path,
    window_size: int = 30,
    step: int = 1,
    rul_limit: int = DEFAULT_RUL_LIMIT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Pipeline completo: carga, limpeza, RUL e janelas para treino e teste.

    Args:
        data_dir: Diretório com os arquivos ``*_FD001.txt``.
        window_size: Comprimento da janela temporal.
        step: Passo entre janelas.
        rul_limit: Truncamento máximo de RUL.

    Returns:
        Tupla ``(x_train, y_train, x_test, y_test, scaler)``.
    """
    raw = load_fd001_raw(data_dir)
    train, test = drop_missing_rows(raw.train, raw.test)
    train, test, rul_df = assign_fd001_column_names(train, test, raw.rul)
    features_train, features_test = drop_unused_sensor_columns(train, test)
    features_train, features_test, scaler = scale_sensor_features(
        features_train, features_test
    )
    rul_train = compute_train_rul(train, limit=rul_limit)
    rul_test = compute_test_rul(test, rul_df, limit=rul_limit)
    x_train, y_train = time_window(
        features_train, rul_train, window_size, step
    )
    x_test, y_test = time_window(features_test, rul_test, window_size, step)
    return x_train, y_train, x_test, y_test, scaler
