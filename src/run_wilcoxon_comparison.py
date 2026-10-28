"""Execute src/comparacao_wilcoxon.ipynb logic and optionally write comparison_2.md."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (  # noqa: E402
    compare_models_wilcoxon,
    load_experiment_results,
    summarize_experiment_results,
)

RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "results"
RESULTS_LSTM = RESULTS_DIR / "lstm_results.csv"
RESULTS_BILSTM = RESULTS_DIR / "bilstm_results.csv"
RESULTS_TRANSFORMER = RESULTS_DIR / "transformer_results.csv"


def main(*, write_markdown: bool = False) -> None:
    missing = [
        path
        for path in (RESULTS_LSTM, RESULTS_BILSTM, RESULTS_TRANSFORMER)
        if not path.is_file()
    ]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Execute src/lstm.ipynb, src/bilstm.ipynb e src/transformer.ipynb "
            f"antes desta comparação. Arquivos ausentes: {names}"
        )

    resultados_lstm = load_experiment_results(RESULTS_LSTM)
    resultados_bilstm = load_experiment_results(RESULTS_BILSTM)
    resultados_transformer = load_experiment_results(RESULTS_TRANSFORMER)

    for label, resultados in (
        ("LSTM", resultados_lstm),
        ("BiLSTM", resultados_bilstm),
        ("Transformer", resultados_transformer),
    ):
        print(f"Resultados {label}".center(40, "-"), flush=True)
        print(f"{resultados.to_string(index=False)}\n", flush=True)
        resumo = summarize_experiment_results(resultados)
        print(f"Resumo {label}".center(40, "-"), flush=True)
        print(f"{resumo}\n", flush=True)

    pairs = (
        ("LSTM vs BiLSTM", resultados_lstm, resultados_bilstm),
        ("LSTM vs Transformer", resultados_lstm, resultados_transformer),
        ("BiLSTM vs Transformer", resultados_bilstm, resultados_transformer),
    )
    for pair_label, left, right in pairs:
        wilcoxon_df = compare_models_wilcoxon(left, right)
        print(f"Teste de Wilcoxon: {pair_label}".center(40, "-"), flush=True)
        for _, row in wilcoxon_df.iterrows():
            metrica = row["metrica"]
            valorp = row["p_value"]
            if row["significativo"]:
                conclusao = "A diferença é estatisticamente significativa"
            else:
                conclusao = "A diferença não é estatisticamente significativa"
            print(f"O p-value para {metrica} é {valorp}. {conclusao}", flush=True)
        print(flush=True)

    if write_markdown:
        from src.write_comparison_2 import main as write_comparison_2  # noqa: WPS433

        write_comparison_2()


if __name__ == "__main__":
    main(write_markdown="--write-md" in sys.argv)
