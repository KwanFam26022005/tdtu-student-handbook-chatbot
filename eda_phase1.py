import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# PATH CONFIG (LOCAL WINDOWS)
# =========================================================
PROCESSED_DIR = r"C:\Users\User\Desktop\datamining\processed"
OUTPUT_EDA_DIR = r"C:\Users\User\Desktop\datamining\eda_results"

os.makedirs(OUTPUT_EDA_DIR, exist_ok=True)

# File paths
chunks_path = os.path.join(PROCESSED_DIR, "chunks.json")
qa_path = os.path.join(PROCESSED_DIR, "qa_all.json")

# =========================================================
# GLOBAL PLOT STYLE
# =========================================================
sns.set_theme(style="whitegrid")

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11

# =========================================================
# 1. CHUNKING ANALYTICS
# =========================================================
if os.path.exists(chunks_path):

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    df_chunks = pd.DataFrame(chunks)

    # Length statistics
    df_chunks["Length"] = df_chunks["text"].apply(len)

    # Metadata extraction success
    df_chunks["Has_Section"] = df_chunks["section"].apply(
        lambda x: bool(str(x).strip()) if pd.notna(x) else False
    )

    print("=" * 60)
    print("SEMANTIC CHUNKING ANALYSIS (PHASE 2B)")
    print("=" * 60)

    print(f"Total chunks               : {len(df_chunks):,}")
    print(f"Average chunk length       : {df_chunks['Length'].mean():.0f} chars")
    print(f"Min length                 : {df_chunks['Length'].min()}")
    print(f"Max length                 : {df_chunks['Length'].max()}")

    metadata_count = df_chunks["Has_Section"].sum()

    print(
        f"Chunks with legal metadata : "
        f"{metadata_count} ({metadata_count / len(df_chunks):.1%})"
    )

    print("=" * 60)

    # -----------------------------------------------------
    # VISUALIZATION
    # -----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Histogram
    sns.histplot(
        data=df_chunks,
        x="Length",
        bins=30,
        kde=True,
        color="#2ecc71",
        ax=axes[0]
    )

    axes[0].set_title(
        "Chunk Length Distribution\n[After Normalization]",
        fontweight="bold"
    )

    axes[0].set_xlabel("Number of Characters")
    axes[0].set_ylabel("Number of Chunks")

    axes[0].axvline(
        x=80,
        color="red",
        linestyle="--",
        label="Min Threshold (80)"
    )

    axes[0].axvline(
        x=3000,
        color="red",
        linestyle="--",
        label="Max Threshold (3000)"
    )

    axes[0].legend()

    # Pie chart
    section_counts = df_chunks["Has_Section"].value_counts()

    axes[1].pie(
        section_counts,
        labels=[
            "Detected Điều/Khoản",
            "Fallback Paragraph"
        ],
        autopct="%1.1f%%",
        colors=["#3498db", "#bdc3c7"],
        startangle=90,
        wedgeprops={"edgecolor": "white"}
    )

    axes[1].set_title(
        "Legal Structure Extraction Rate",
        fontweight="bold"
    )

    plt.tight_layout()

    chunk_fig_path = os.path.join(
        OUTPUT_EDA_DIR,
        "chunking_analysis.png"
    )

    plt.savefig(chunk_fig_path)
    plt.show()

    print(f"[OK] Saved: {chunk_fig_path}")

else:
    print(f"[WARNING] File not found: {chunks_path}")

# =========================================================
# 2. QA DATASET ANALYSIS
# =========================================================
if os.path.exists(qa_path):

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    df_qa = pd.DataFrame(qa_data)

    # Length analysis
    df_qa["Q_Length"] = df_qa["question"].apply(
        lambda x: len(str(x).split())
    )

    df_qa["A_Length"] = df_qa["answer"].apply(
        lambda x: len(str(x).split())
    )

    print("\n" + "=" * 60)
    print("QA DATASET ANALYSIS (PHASE 2D)")
    print("=" * 60)

    print(f"Total QA pairs             : {len(df_qa):,}")

    print(
        f"Average question length    : "
        f"{df_qa['Q_Length'].mean():.0f} words"
    )

    print(
        f"Average answer length      : "
        f"{df_qa['A_Length'].mean():.0f} words"
    )

    print("=" * 60)

    # -----------------------------------------------------
    # VISUALIZATION
    # -----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # QA type distribution
    type_counts = df_qa["type"].value_counts()

    sns.barplot(
        x=type_counts.index,
        y=type_counts.values,
        hue=type_counts.index,
        palette="Set2",
        legend=False,
        ax=axes[0]
    )

    axes[0].set_title(
        "QA Pair Type Distribution",
        fontweight="bold"
    )

    axes[0].set_ylabel("Count")
    axes[0].set_xlabel("Question Type")

    for i, v in enumerate(type_counts.values):
        axes[0].text(
            i,
            v + (max(type_counts.values) * 0.01),
            str(v),
            ha="center",
            fontweight="bold"
        )

    # Generated source pie chart
    source_counts = df_qa["generated_by"].value_counts()

    axes[1].pie(
        source_counts,
        labels=source_counts.index,
        autopct="%1.1f%%",
        colors=["#9b59b6", "#e67e22"],
        startangle=90,
        wedgeprops={"edgecolor": "white"}
    )

    axes[1].set_title(
        "QA Generation Sources",
        fontweight="bold"
    )

    plt.tight_layout()

    qa_fig_path = os.path.join(
        OUTPUT_EDA_DIR,
        "qa_analysis.png"
    )

    plt.savefig(qa_fig_path)
    plt.show()

    print(f"[OK] Saved: {qa_fig_path}")

else:
    print(f"[WARNING] File not found: {qa_path}")

# =========================================================
# FINISH
# =========================================================
print("\n" + "=" * 60)
print(f"EDA results saved to:")
print(OUTPUT_EDA_DIR)
print("=" * 60)