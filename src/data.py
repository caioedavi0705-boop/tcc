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


def _resolve_fd001_file_paths(base: Path) -> tuple[Path, Path, Path]:
    """Resolve caminhos FD001 (flat ou ``training/``, ``test/``, ``result/``)."""
    flat_train = base / "train_FD001.txt"
    flat_test = base / "test_FD001.txt"
    flat_rul = base / "RUL_FD001.txt"
    if flat_train.is_file() and flat_test.is_file() and flat_rul.is_file():
        return flat_train, flat_test, flat_rul
    return (
        base / "training" / "train_FD001.txt",
        base / "test" / "test_FD001.txt",
        base / "result" / "RUL_FD001.txt",
    )


def load_fd001_raw(data_dir: str | Path) -> FD001RawData:
    """Carrega os arquivos de treino, teste e RUL do FD001.

    Args:
        data_dir: Diretório base com os arquivos FD001 (mesmo nível ou subpastas
            ``training/``, ``test/`` e ``result/``).

    Returns:
        Tupla nomeada com os três DataFrames lidos (sem cabeçalho).
    """
    base = Path(data_dir)
    train_path, test_path, rul_path = _resolve_fd001_file_paths(base)
    read_kwargs = {"sep": r"\s+", "header": None}
    train = pd.read_csv(train_path, **read_kwargs)
    test = pd.read_csv(test_path, **read_kwargs)
    rul = pd.read_csv(rul_path, **read_kwargs)
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
    """Remove colunas de sensores não utilizadas no FD001.

    Args:
        train: DataFrame de treino com cabeçalho.
        test: DataFrame de teste com cabeçalho.
        columns_to_drop: Colunas a remover; usa ``COLUMNS_TO_DROP`` se omitida.

    Returns:
        Par ``(train, test)`` sem as colunas descartadas.
    """
    drop_cols = columns_to_drop or COLUMNS_TO_DROP
    return (
        train.drop(columns=drop_cols).copy(),
        test.drop(columns=drop_cols).copy(),
    )


def compute_train_rul(
    train: pd.DataFrame,
    limit: int = DEFAULT_RUL_LIMIT,
    id_column: str = ID_COLUMN,
    cycle_column: str = CYCLE_COLUMN,
) -> list[int]:
    """Calcula RUL por ciclo no treino (truncamento piecewise).

    Args:
        train: DataFrame de treino com ID e número de ciclos.
        limit: Teto de RUL (padrão 130 ciclos).
        id_column: Coluna de identificação do motor.
        cycle_column: Coluna de ciclo operacional.

    Returns:
        Lista de RUL alinhada à ordem das linhas por motor.
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
                rul_values.append(
                    int(motor_max - engine[cycle_column].iloc[cycle_index])
                )
    return [int(value) for value in rul_values]


def compute_test_rul(
    test: pd.DataFrame,
    rul_remaining: pd.DataFrame,
    limit: int = DEFAULT_RUL_LIMIT,
    id_column: str = ID_COLUMN,
    cycle_column: str = CYCLE_COLUMN,
    rul_column: str = RUL_COLUMN,
) -> list[int]:
    """Calcula RUL por ciclo no teste usando o RUL verdadeiro final.

    Args:
        test: DataFrame de teste com ID e ciclos.
        rul_remaining: RUL ao fim da série de teste por motor.
        limit: Teto de RUL.
        id_column: Coluna de ID.
        cycle_column: Coluna de ciclo.
        rul_column: Coluna de RUL em ``rul_remaining``.

    Returns:
        Lista de RUL alinhada às linhas de ``test``.
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
    from src.features import create_fd001_windowed_features

    raw = load_fd001_raw(data_dir)
    train, test = drop_missing_rows(raw.train, raw.test)
    train, test, rul_df = assign_fd001_column_names(train, test, raw.rul)
    rul_train = compute_train_rul(train, limit=rul_limit)
    rul_test = compute_test_rul(test, rul_df, limit=rul_limit)
    train, test = drop_unused_sensor_columns(train, test)
    windowed = create_fd001_windowed_features(
        train,
        test,
        rul_train,
        rul_test,
        window_size=window_size,
        step=step,
    )
    return (
        windowed.x_train,
        windowed.y_train,
        windowed.x_test,
        windowed.y_test,
        windowed.scaler,
    )
