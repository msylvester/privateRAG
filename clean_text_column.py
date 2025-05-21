
import pandas as pd
import re
import argparse
import os

def clean_csv_text_column(input_file, output_file=None, text_column="text"):
    """
    Clean only the text column in a CSV file for RAG processing.

    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the cleaned CSV file (defaults to input_file with '_cleaned' suffix)
        text_column: Name of the text column to clean (defaults to "text")
    """
    # Default output filename if not provided
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_cleaned{ext}"

    # Load the CSV file
    try:
        df = pd.read_csv(input_file, quoting=1, engine='python', on_bad_lines='skip')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    # Check if text column exists
    if text_column not in df.columns:
        print(f"Column '{text_column}' not found in CSV")
        return None

    # Clean only the text column
    df[text_column] = df[text_column].fillna("")
    df[text_column] = df[text_column].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())

    # Save the cleaned CSV
    df.to_csv(output_file, index=False, quoting=1)
    print(f"Saved cleaned CSV to {output_file}")

    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean text column in a CSV file for RAG")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("--output", "-o", help="Path to save the cleaned CSV file")
    parser.add_argument("--column", "-c", default="text", help="Name of the text column to clean")

    args = parser.parse_args()
    clean_csv_text_column(args.input_file, args.output, args.column)
