
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_path = Path("dataset/Sample - Superstore_2024.xlsx")

df = pd.read_excel(file_path, sheet_name="Orders")

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nColumn data types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDescriptive statistics:")
print(df.describe().T)


print("\n========== DATA QUALITY ANALYSIS ==========")
print("\nMissing values in each column:")
print(df.isnull().sum())
missing_percentage = (df.isnull().sum() / len(df)) * 100

print("\nMissing value percentage:")
print(missing_percentage.round(2))
duplicate_count = df.duplicated().sum()

print("\nNumber of duplicate rows:")
print(duplicate_count)
print("\nCurrent data types:")
print(df.dtypes)
for column in ["Order Date", "Ship Date"]:
    if column in df.columns:
        df[column] = pd.to_datetime(
            df[column], errors="coerce"
        )

print("\nUpdated date column types:")
for column in ["Order Date", "Ship Date"]:
    if column in df.columns:
        print(column, ":", df[column].dtype)
quality_report = pd.DataFrame({
    "Missing Values": df.isnull().sum(),
    "Missing Percentage": (
        df.isnull().sum() / len(df) * 100
    ).round(2),
    "Data Type": df.dtypes.astype(str)
})

Path("outputs").mkdir(parents=True, exist_ok=True)

quality_report.to_csv(
    "outputs/data_quality_report.csv"
)

print("\nData quality report saved successfully!")

print("\n========== DESCRIPTIVE STATISTICS ==========")
columns_to_analyze = [
    col for col in ["Sales", "Profit", "Quantity", "Discount"]
    if col in df.columns
]
statistics = df[columns_to_analyze].describe().T
statistics["median"] = df[columns_to_analyze].median()

print(statistics.round(2))
statistics.to_csv("outputs/descriptive_statistics.csv")

print("\nDescriptive statistics saved successfully!")

print("\n========== OUTLIER ANALYSIS ==========")

outlier_results = []

for column in columns_to_analyze:
    values = df[column].dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    outliers = values[
        (values < lower_limit) |
        (values > upper_limit)
    ]

    outlier_results.append({
        "Column": column,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower Limit": lower_limit,
        "Upper Limit": upper_limit,
        "Outlier Count": len(outliers)
    })

outlier_report = pd.DataFrame(outlier_results)

print(outlier_report.round(2))

outlier_report.to_csv(
    "outputs/outlier_summary.csv",
    index=False
)

print("\nOutlier report saved successfully!")

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

for column in ["Sales", "Profit"]:
    if column in df.columns:
        plt.figure(figsize=(9, 4))

        sns.boxplot(x=df[column])

        plt.title(f"Distribution and Potential Outliers: {column}")
        plt.xlabel(column)

        plt.tight_layout()
        plt.savefig(
            f"outputs/{column.lower()}_boxplot.png",
            dpi=200
        )

        plt.close()

print("\nBox plots saved successfully!")

print("\n========== CATEGORY PERFORMANCE ==========")

category_summary = (
    df.groupby("Category")
    .agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
        Average_Sales=("Sales", "mean"),
        Average_Profit=("Profit", "mean")
    )
    .sort_values("Total_Sales", ascending=False)
)

print(category_summary.round(2))
category_summary.to_csv(
    "outputs/category_summary.csv"
)
highest_sales_category = category_summary["Total_Sales"].idxmax()
highest_profit_category = category_summary["Total_Profit"].idxmax()

print("\nCategory with highest total sales:",
      highest_sales_category)

print("Category with highest total profit:",
      highest_profit_category)

print("\nCategory analysis completed!")

print("\n========== REGIONAL PERFORMANCE ==========")

region_summary = (
    df.groupby("Region")
    .agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum")
    )
    .sort_values("Total_Sales", ascending=False)
)

print(region_summary.round(2))

region_summary.to_csv(
    "outputs/region_summary.csv"
)

best_sales_region = region_summary["Total_Sales"].idxmax()
best_profit_region = region_summary["Total_Profit"].idxmax()

print("\nRegion with highest sales:", best_sales_region)
print("Region with highest profit:", best_profit_region)

print("\nRegional analysis completed!")

print("\n========== MONTHLY SALES TREND ==========")
df["Order Date"] = pd.to_datetime(
    df["Order Date"], errors="coerce"
)

monthly_sales = (
    df.dropna(subset=["Order Date"])
    .set_index("Order Date")["Sales"]
    .resample("MS")
    .sum()
)

print(monthly_sales)

monthly_sales.to_csv(
    "outputs/monthly_sales.csv"
)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
monthly_sales.plot()

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.tight_layout()

plt.savefig(
    "outputs/monthly_sales_trend.png",
    dpi=200
)

plt.close()

print("\nMonthly sales trend saved successfully!")

import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
Path("outputs").mkdir(parents=True, exist_ok=True)

print("\n========== CORRELATION ANALYSIS ==========")
numeric_columns = [
    col for col in ["Sales", "Profit", "Quantity", "Discount"]
    if col in df.columns
]
correlation_matrix = df[numeric_columns].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.round(3))
correlation_matrix.to_csv(
    "outputs/correlation_matrix.csv"
)
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)

plt.title("Correlation Between Sales, Profit, Quantity and Discount")
plt.tight_layout()

plt.savefig(
    "outputs/correlation_heatmap.png",
    dpi=200
)

plt.close()

print("\nCorrelation heatmap saved successfully!")

print("\n========== DISCOUNT VS PROFIT ==========")

if {"Discount", "Profit"}.issubset(df.columns):

    discount_profit_corr = df["Discount"].corr(df["Profit"])

    print(
        "Correlation between Discount and Profit:",
        round(discount_profit_corr, 3)
    )

    plt.figure(figsize=(9, 5))

    sns.scatterplot(
        data=df,
        x="Discount",
        y="Profit",
        alpha=0.5
    )

    plt.title("Relationship Between Discount and Profit")
    plt.xlabel("Discount")
    plt.ylabel("Profit")
    plt.tight_layout()

    plt.savefig(
        "outputs/discount_vs_profit.png",
        dpi=200
    )

    plt.close()

    print("Discount-profit scatter plot saved successfully!")

else:
    print("Required columns are missing.")

from scipy.stats import pearsonr, f_oneway

print("\n========== HYPOTHESIS TESTING ==========")

Path("outputs").mkdir(parents=True, exist_ok=True)

with open(
    "outputs/hypothesis_tests.txt",
    "w",
    encoding="utf-8"
) as report:
    test_data = df[["Discount", "Profit"]].dropna()

    if (
        len(test_data) >= 3
        and test_data["Discount"].nunique() > 1
        and test_data["Profit"].nunique() > 1
    ):
        r, p_value = pearsonr(
            test_data["Discount"],
            test_data["Profit"]
        )

        report.write("TEST 1: Discount and Profit\n")
        report.write(
            "H0: There is no linear correlation.\n"
        )
        report.write(
            "H1: There is a linear correlation.\n"
        )
        report.write(f"Pearson correlation: {r:.4f}\n")
        report.write(f"P-value: {p_value:.6f}\n")

        if p_value < 0.05:
            report.write(
                "Decision: Reject the null hypothesis.\n\n"
            )
        else:
            report.write(
                "Decision: Fail to reject the null hypothesis.\n\n"
            )

        print("Discount-profit correlation:", round(r, 4))
        print("Correlation p-value:", round(p_value, 6))
    if {"Category", "Profit"}.issubset(df.columns):
        category_groups = [
            group["Profit"].dropna().to_numpy()
            for _, group in df.groupby(
                df["Category"].astype("string").str.strip().str.casefold()
            )
        ]

        category_groups = [
            group for group in category_groups
            if len(group) >= 2
        ]

        if len(category_groups) >= 2:
            f_stat, anova_p = f_oneway(*category_groups)

            report.write(
                "TEST 2: Average profit across categories\n"
            )
            report.write(
                "H0: All category mean profits are equal.\n"
            )
            report.write(
                "H1: At least one category mean differs.\n"
            )
            report.write(f"F-statistic: {f_stat:.4f}\n")
            report.write(f"P-value: {anova_p:.6f}\n")

            if anova_p < 0.05:
                report.write(
                    "Decision: Reject the null hypothesis.\n"
                )
            else:
                report.write(
                    "Decision: Fail to reject the null hypothesis.\n"
                )

            print("Category profit ANOVA F-statistic:", round(f_stat, 4))
            print("Category profit ANOVA p-value:", round(anova_p, 6))

print("\nHypothesis test report saved in outputs!")

print("\n========== FINAL EDA REPORT ==========")

Path("outputs").mkdir(parents=True, exist_ok=True)

with open(
    "outputs/final_eda_report.txt",
    "w",
    encoding="utf-8"
) as report:

    report.write("CODEALPHA TASK 2: EXPLORATORY DATA ANALYSIS\n")
    report.write("=" * 48 + "\n\n")

    report.write(f"Total rows: {len(df)}\n")
    report.write(f"Total columns: {len(df.columns)}\n")
    report.write(
        f"Duplicate rows: {df.duplicated().sum()}\n"
    )
    report.write(
        f"Missing cells: {df.isnull().sum().sum()}\n\n"
    )

    if "Sales" in df.columns:
        report.write(
            f"Total sales: {df['Sales'].sum():,.2f}\n"
        )

    if "Profit" in df.columns:
        report.write(
            f"Total profit: {df['Profit'].sum():,.2f}\n"
        )

    if {"Category", "Sales", "Profit"}.issubset(df.columns):
        category_results = df.groupby(
            df["Category"].astype("string").str.strip().str.casefold()
        ).agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum")
        )

        report.write("\nCATEGORY PERFORMANCE\n")
        report.write(category_results.to_string() + "\n")

    if {"Region", "Sales", "Profit"}.issubset(df.columns):
        region_results = df.groupby("Region").agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum")
        )

        report.write("\nREGIONAL PERFORMANCE\n")
        report.write(region_results.to_string() + "\n")

    report.write(
        "\nSee the outputs folder for detailed statistics, "
        "charts, and hypothesis tests.\n"
    )

print("Final EDA report saved successfully!")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(
    style="whitegrid",
    context="notebook",
    palette="deep"
)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 220
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.figsize"] = (10, 6)
