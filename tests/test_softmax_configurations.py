#!/usr/bin/env python3
"""
Test script to run eval_softmax_sink with multiple configurations
and collect statistics for comparison.
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
import argparse
import time


def run_softmax_test(activation_type, outlier_threshold, output_base_dir, args):
    """Run a single test configuration."""
    output_dir = Path(output_base_dir) / f"{activation_type}_thresh{outlier_threshold}"

    # Modify the eval_softmax_sink.py temporarily to use different activation_type
    cmd = [
        "python",
        "eval_softmax_sink.py",
        "--output_dir",
        str(output_dir),
        "--outlier_threshold",
        str(outlier_threshold),
        "--calib_samples",
        str(args.calib_samples),
        "--ppl_seqlen",
        str(args.ppl_seqlen),
    ]

    if args.no_prefix:
        cmd.append("--no_prefix")

    print(f"\n{'='*60}")
    print(f"Running test: activation={activation_type}, threshold={outlier_threshold}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}")

    # Create a temporary modified version of eval_softmax_sink.py
    with open("eval_softmax_sink.py", "r") as f:
        original_content = f.read()

    # Replace activation_type in the file
    modified_content = original_content.replace(
        'activation_type="down_proj"', f'activation_type="{activation_type}"'
    )

    with open("eval_softmax_sink_temp.py", "w") as f:
        f.write(modified_content)

    # Run the modified script
    cmd[1] = "eval_softmax_sink_temp.py"

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse results from JSON output
        json_file = output_dir / "per_token_statistics.json"
        if json_file.exists():
            with open(json_file, "r") as f:
                data = json.load(f)
                stats = data.get("global_stats", {})

                return {
                    "activation_type": activation_type,
                    "outlier_threshold": outlier_threshold,
                    "global_prefix_max": stats.get("global_prefix_max", None),
                    "global_postfix_max": stats.get("global_postfix_max", None),
                    "global_prefix_min": stats.get("global_prefix_min", None),
                    "global_postfix_min": stats.get("global_postfix_min", None),
                    "global_max_all_positions": stats.get(
                        "global_max_all_positions", None
                    ),
                    "global_min_all_positions": stats.get(
                        "global_min_all_positions", None
                    ),
                    "max_sequence_prefix_ratio": stats.get(
                        "max_sequence_prefix_ratio", None
                    ),
                    "mean_attention_ratio": stats.get("mean_attention_ratio", None),
                    "max_max_difference": stats.get("max_max_difference", None),
                }
        else:
            print(f"Warning: JSON output not found at {json_file}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Error running test: {e}")
        print(f"Stderr: {e.stderr}")
        return None
    finally:
        # Clean up temp file
        Path("eval_softmax_sink_temp.py").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Test multiple softmax sink configurations"
    )
    parser.add_argument("--output_base_dir", default="./log/softmax_tests", type=str)
    parser.add_argument("--calib_samples", default=64, type=int)
    parser.add_argument("--ppl_seqlen", default=512, type=int)
    parser.add_argument(
        "--no_prefix", action="store_true", help="Run without prefix tokens"
    )
    args = parser.parse_args()

    # Define test configurations
    # Based on the code, valid activation types are: 'hidden_state', 'down_proj', 'q_k_up_gate'

    # For faster testing, you can reduce these:
    activation_types = ["hidden_state"]
    # activation_types = ["hidden_state", "down_proj", "q_k_up_gate"]

    outlier_thresholds = [8]
    # outlier_thresholds = [6,8,10]

    results = []

    # Run tests for different configurations
    for activation_type in activation_types:
        for threshold in outlier_thresholds:
            print(f"\nTesting {activation_type} with threshold {threshold}...")
            result = run_softmax_test(
                activation_type, threshold, args.output_base_dir, args
            )
            if result:
                results.append(result)
                print(f"✓ Test completed successfully")
            else:
                print(f"✗ Test failed or incomplete")

            # Small delay between tests
            time.sleep(1)

    # Create results DataFrame
    if results:
        df = pd.DataFrame(results)

        # Save full results
        output_file = Path(args.output_base_dir) / "test_results.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_file}")

        # Display summary statistics
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")

        # Group by activation type
        print("\n--- By Activation Type ---")
        for act_type in df["activation_type"].unique():
            act_df = df[df["activation_type"] == act_type]
            print(f"\n{act_type}:")
            print(
                f"  Global Prefix Max:  min={act_df['global_prefix_max'].min():.3f}, "
                f"max={act_df['global_prefix_max'].max():.3f}, "
                f"mean={act_df['global_prefix_max'].mean():.3f}"
            )
            print(
                f"  Global Postfix Max: min={act_df['global_postfix_max'].min():.3f}, "
                f"max={act_df['global_postfix_max'].max():.3f}, "
                f"mean={act_df['global_postfix_max'].mean():.3f}"
            )
            if 'global_prefix_min' in act_df.columns and not act_df['global_prefix_min'].isna().all():
                print(
                    f"  Global Prefix Min:  min={act_df['global_prefix_min'].min():.3f}, "
                    f"max={act_df['global_prefix_min'].max():.3f}, "
                    f"mean={act_df['global_prefix_min'].mean():.3f}"
                )
            if 'global_postfix_min' in act_df.columns and not act_df['global_postfix_min'].isna().all():
                print(
                    f"  Global Postfix Min: min={act_df['global_postfix_min'].min():.3f}, "
                    f"max={act_df['global_postfix_min'].max():.3f}, "
                    f"mean={act_df['global_postfix_min'].mean():.3f}"
                )
            if "global_max_all_positions" in act_df.columns:
                if not act_df["global_max_all_positions"].isna().all():
                    print(
                        f"  Global Max All:     {act_df['global_max_all_positions'].max():.3f}"
                    )
            if "global_min_all_positions" in act_df.columns:
                if not act_df["global_min_all_positions"].isna().all():
                    print(
                        f"  Global Min All:     {act_df['global_min_all_positions'].min():.3f}"
                    )
            print(
                f"  Mean Attention Ratio: min={act_df['mean_attention_ratio'].min():.3f}, "
                f"max={act_df['mean_attention_ratio'].max():.3f}, "
                f"mean={act_df['mean_attention_ratio'].mean():.3f}"
            )

        # Group by threshold
        print("\n--- By Outlier Threshold ---")
        for threshold in df["outlier_threshold"].unique():
            thresh_df = df[df["outlier_threshold"] == threshold]
            print(f"\nThreshold {threshold}:")
            print(
                f"  Global Prefix Max:  min={thresh_df['global_prefix_max'].min():.3f}, "
                f"max={thresh_df['global_prefix_max'].max():.3f}, "
                f"mean={thresh_df['global_prefix_max'].mean():.3f}"
            )
            print(
                f"  Global Postfix Max: min={thresh_df['global_postfix_max'].min():.3f}, "
                f"max={thresh_df['global_postfix_max'].max():.3f}, "
                f"mean={thresh_df['global_postfix_max'].mean():.3f}"
            )
            print(
                f"  Mean Attention Ratio: min={thresh_df['mean_attention_ratio'].min():.3f}, "
                f"max={thresh_df['mean_attention_ratio'].max():.3f}, "
                f"mean={thresh_df['mean_attention_ratio'].mean():.3f}"
            )

        # Find best configurations
        print(f"\n{'='*60}")
        print("BEST CONFIGURATIONS")
        print(f"{'='*60}")

        print("\nHighest Mean Attention Ratio:")
        best_ratio = df.loc[df["mean_attention_ratio"].idxmax()]
        print(f"  Activation: {best_ratio['activation_type']}")
        print(f"  Threshold: {best_ratio['outlier_threshold']}")
        print(f"  Attention Ratio: {best_ratio['mean_attention_ratio']:.3f}")

        print("\nLowest Global Prefix Max (strongest sink):")
        best_sink = df.loc[df["global_prefix_max"].idxmin()]
        print(f"  Activation: {best_sink['activation_type']}")
        print(f"  Threshold: {best_sink['outlier_threshold']}")
        print(f"  Global Prefix Max: {best_sink['global_prefix_max']:.3f}")

        print("\nHighest Max Sequence/Prefix Ratio:")
        if (
            "max_sequence_prefix_ratio" in df.columns
            and not df["max_sequence_prefix_ratio"].isna().all()
        ):
            best_ratio = df.loc[df["max_sequence_prefix_ratio"].idxmax()]
            print(f"  Activation: {best_ratio['activation_type']}")
            print(f"  Threshold: {best_ratio['outlier_threshold']}")
            print(f"  Ratio: {best_ratio['max_sequence_prefix_ratio']:.3f}")

        # Display full table
        print(f"\n{'='*60}")
        print("FULL RESULTS TABLE")
        print(f"{'='*60}")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option(
            "display.float_format", lambda x: f"{x:.3f}" if pd.notnull(x) else "NaN"
        )
        print(df.to_string(index=False))

    else:
        print("\nNo successful test results collected.")


if __name__ == "__main__":
    main()
