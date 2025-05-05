import argparse
import re
import sys
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt


def extract_ppl(path: str) -> Dict[int, float]:

    step_pattern = re.compile(r"Step:\s*(\d+),")
    ppl_pattern = re.compile(r"ppl:\s*([0-9]+\.?[0-9]*)")
    results: Dict[int, float] = {}
    current_step = None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            m_step = step_pattern.search(line)
            if m_step:
                current_step = int(m_step.group(1))
            m_ppl = ppl_pattern.search(line)
            if m_ppl and current_step is not None:
                results[current_step] = float(m_ppl.group(1))
                current_step = None

    return results


def plot_ppl(df: pd.DataFrame, output_path: str) -> None:

    plt.figure()
    for model in ['Baseline', 'Prenorm', 'Postnorm']:
        plt.plot(df['Validation ppl'], df[model], label=model)
    plt.xlabel('Training Step')
    plt.ylabel('Perplexity (ppl)')
    plt.title('Validation Perplexity over Training Steps within 10 Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract validation ppl and generate CSV and PDF plot."
    )
    parser.add_argument('baseline_log', help='Baseline train.log path')
    parser.add_argument('prenorm_log', help='Pre-norm train.log path')
    parser.add_argument('postnorm_log', help='Post-norm train.log path')
    parser.add_argument('--csv_output', required=True,
                        help='Path to save the CSV table')
    parser.add_argument('--pdf_output', required=True,
                        help='Path to save the PDF plot')
    args = parser.parse_args()


    baseline = extract_ppl(args.baseline_log)
    prenorm = extract_ppl(args.prenorm_log)
    postnorm = extract_ppl(args.postnorm_log)

    all_steps = sorted(set(baseline) | set(prenorm) | set(postnorm))
    df = pd.DataFrame({
        'Validation ppl': all_steps,
        'Baseline':    [baseline.get(s) for s in all_steps],
        'Prenorm':     [prenorm.get(s) for s in all_steps],
        'Postnorm':    [postnorm.get(s) for s in all_steps],
    })

    df.to_csv(args.csv_output, index=False)
    print(f"Saved CSV to {args.csv_output}")

    plot_ppl(df, args.pdf_output)

if __name__ == '__main__':
    main()
