import pandas as pd
import io

df = pd.read_csv('2009.csv/2009.csv', nrows=100000)

with open('data_exploration.md', 'w') as f:
    f.write("# Dataset Exploration (Sample: 100,000 rows)\n\n")
    f.write("## Shape\n")
    f.write(f"- Rows: {df.shape[0]}\n")
    f.write(f"- Columns: {df.shape[1]}\n\n")

    f.write("## Columns and Missing Values\n")
    f.write("A summary of missing values for the sampled data:\n\n")
    f.write("| Column Name | Missing Values count | Missing Values % | Data Type |\n")
    f.write("| --- | --- | --- | --- |\n")
    for col in df.columns:
        missing = df[col].isnull().sum()
        missing_pct = missing / len(df) * 100
        dtype = df[col].dtype
        f.write(f"| {col} | {missing} | {missing_pct:.2f}% | {dtype} |\n")

    f.write("\n## Descriptive Statistics\n")
    f.write("```text\n")
    f.write(df.describe().to_string())
    f.write("\n```\n")
    
    f.write("\n## First 5 Rows\n")
    f.write("```text\n")
    f.write(df.head().to_string())
    f.write("\n```\n")

print("Exploration finished. Output written to data_exploration.md")
