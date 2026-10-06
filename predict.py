"""Recommend crops for a soil/climate sample using the trained model.

Example:
  python predict.py --N 90 --P 42 --K 43 --temperature 21 --humidity 82 --ph 6.5 --rainfall 203
"""
import argparse

import crop_model as cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/crop_model.pt")
    ap.add_argument("--csv", default="data/Crop_recommendation.csv", help="used for the soil report")
    ap.add_argument("--top-k", type=int, default=3)
    for f in cm.FEATURES:
        ap.add_argument(f"--{f}", type=float, required=True)
    args = ap.parse_args()

    sample = {f: getattr(args, f) for f in cm.FEATURES}
    model, bundle = cm.load_bundle(args.model)
    results = cm.recommend(model, bundle, sample, top_k=args.top_k)

    print("Top recommendations:")
    for crop, p in results:
        print(f"  {crop:<12} {p * 100:5.1f}%")

    print("\nSoil analysis for the top crop:")
    for line in cm.soil_report(sample, cm.load_dataframe(args.csv), results[0][0]):
        print(" -", line)


if __name__ == "__main__":
    main()
