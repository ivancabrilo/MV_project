import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_SUMMARY = "model_predictions/test_video2/summary.csv"


def load_rows(summary_path):
    with open(summary_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def usable_rows(rows):
    usable = []
    skipped = Counter()

    for row in rows:
        actual = (row.get("actual_label") or "").strip()
        predicted = (row.get("model_prediction") or "").strip()

        if not actual:
            skipped["missing_actual_label"] += 1
            continue
        if not predicted:
            skipped["missing_model_prediction"] += 1
            continue
        if actual == "unlabeled":
            skipped["unlabeled_actual"] += 1
            continue

        usable.append({**row, "actual_label": actual, "model_prediction": predicted})

    return usable, skipped


def safe_div(num, den):
    return num / den if den else 0.0


def build_confusion_matrix(rows, labels):
    matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for row in rows:
        matrix[row["actual_label"]][row["model_prediction"]] += 1
    return matrix


def class_metrics(matrix, labels):
    metrics = {}
    total = sum(sum(row.values()) for row in matrix.values())

    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in labels if actual != label)
        fn = sum(matrix[label][predicted] for predicted in labels if predicted != label)
        tn = total - tp - fp - fn

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        support = tp + fn

        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    return metrics


def aggregate_metrics(per_class):
    labels = list(per_class)
    total_support = sum(per_class[label]["support"] for label in labels)

    macro = {
        key: safe_div(sum(per_class[label][key] for label in labels), len(labels))
        for key in ["precision", "recall", "f1"]
    }
    weighted = {
        key: safe_div(
            sum(per_class[label][key] * per_class[label]["support"] for label in labels),
            total_support,
        )
        for key in ["precision", "recall", "f1"]
    }
    return macro, weighted


def binary_punch_metrics(rows):
    converted = []
    for row in rows:
        actual = "no_punch" if row["actual_label"] == "no_punch" else "punch"
        predicted = "no_punch" if row["model_prediction"] == "no_punch" else "punch"
        converted.append({"actual_label": actual, "model_prediction": predicted})

    labels = ["punch", "no_punch"]
    matrix = build_confusion_matrix(converted, labels)
    metrics = class_metrics(matrix, labels)

    return {
        "labels": labels,
        "confusion_matrix": matrix,
        "punch_precision": metrics["punch"]["precision"],
        "punch_recall": metrics["punch"]["recall"],
        "punch_f1": metrics["punch"]["f1"],
        "no_punch_precision": metrics["no_punch"]["precision"],
        "no_punch_recall": metrics["no_punch"]["recall"],
        "no_punch_f1": metrics["no_punch"]["f1"],
    }


def write_confusion_csv(matrix, labels, output_path):
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted", *labels])
        for actual in labels:
            writer.writerow([actual, *[matrix[actual][predicted] for predicted in labels]])


def write_class_metrics_csv(per_class, output_path):
    fields = ["label", "precision", "recall", "f1", "support", "tp", "fp", "fn", "tn"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label, metrics in per_class.items():
            writer.writerow({"label": label, **metrics})


def print_report(results):
    print("\nOverall")
    print(f"  rows in summary.csv: {results['row_count']}")
    print(f"  evaluated rows:      {results['evaluated_count']}")
    print(f"  accuracy:            {results['accuracy']:.4f}")
    print(f"  macro F1:            {results['macro_avg']['f1']:.4f}")
    print(f"  weighted F1:         {results['weighted_avg']['f1']:.4f}")

    print("\nPer-class metrics")
    header = f"{'label':<16} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>8}"
    print(header)
    print("-" * len(header))
    for label, metrics in results["per_class"].items():
        print(
            f"{label:<16} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f} "
            f"{metrics['support']:>8}"
        )

    print("\nPunch vs no_punch")
    binary = results["binary_punch_vs_no_punch"]
    print(f"  punch precision:     {binary['punch_precision']:.4f}")
    print(f"  punch recall:        {binary['punch_recall']:.4f}")
    print(f"  punch F1:            {binary['punch_f1']:.4f}")
    print(f"  no_punch precision:  {binary['no_punch_precision']:.4f}")
    print(f"  no_punch recall:     {binary['no_punch_recall']:.4f}")
    print(f"  no_punch F1:         {binary['no_punch_f1']:.4f}")

    if results["skipped_rows"]:
        print("\nSkipped rows")
        for reason, count in results["skipped_rows"].items():
            print(f"  {reason}: {count}")


def evaluate(summary_path, output_dir):
    rows = load_rows(summary_path)
    rows, skipped = usable_rows(rows)

    if not rows:
        raise RuntimeError("No usable rows found. Make sure summary.csv has model_prediction and actual_label columns.")

    labels = sorted({row["actual_label"] for row in rows} | {row["model_prediction"] for row in rows})
    matrix = build_confusion_matrix(rows, labels)
    per_class = class_metrics(matrix, labels)
    macro_avg, weighted_avg = aggregate_metrics(per_class)

    correct = sum(1 for row in rows if row["actual_label"] == row["model_prediction"])
    accuracy = safe_div(correct, len(rows))

    results = {
        "summary_path": str(summary_path),
        "row_count": len(load_rows(summary_path)),
        "evaluated_count": len(rows),
        "correct_count": correct,
        "accuracy": accuracy,
        "labels": labels,
        "per_class": per_class,
        "macro_avg": macro_avg,
        "weighted_avg": weighted_avg,
        "confusion_matrix": matrix,
        "binary_punch_vs_no_punch": binary_punch_metrics(rows),
        "skipped_rows": dict(skipped),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    confusion_path = output_dir / "confusion_matrix.csv"
    class_metrics_path = output_dir / "class_metrics.csv"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_confusion_csv(matrix, labels, confusion_path)
    write_class_metrics_csv(per_class, class_metrics_path)

    print_report(results)
    print("\nWrote")
    print(f"  {metrics_path}")
    print(f"  {confusion_path}")
    print(f"  {class_metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate punch model performance from predict_punches summary.csv.")
    parser.add_argument(
        "summary_csv",
        nargs="?",
        default=DEFAULT_SUMMARY,
        help="Path to model_predictions/<video>/summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for metrics.json, confusion_matrix.csv, and class_metrics.csv. Defaults next to summary.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary_path = Path(args.summary_csv)
    output_dir = Path(args.output_dir) if args.output_dir else summary_path.parent
    evaluate(summary_path, output_dir)
