import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# === STEP 1: Set up file paths dynamically ===
base_dir = os.getcwd()
input_filename = "spotify_tracks.csv"
csv_path = os.path.join(base_dir, "data", input_filename)
output_dir = os.path.join(base_dir, "data", "output")
os.makedirs(output_dir, exist_ok=True)

# === STEP 2: Load and clean the data ===
df = pd.read_csv(csv_path)
df_cleaned = df.dropna()
columns_to_drop = ['track_id', 'track_name', 'album_name']
df_cleaned = df_cleaned.drop(columns=[col for col in columns_to_drop if col in df_cleaned.columns])
df_cleaned['energy_loudness_ratio'] = df_cleaned['energy'] / df_cleaned['loudness'].abs()
df_cleaned['duration_min'] = df_cleaned['duration_ms'] / 60000
df_cleaned = df_cleaned.drop_duplicates()

print("Most common musical key:", df_cleaned['key'].mode()[0])

# === STEP 3: Correlation heatmap ===
numeric_df = df_cleaned.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("🎼 Correlation Heatmap of Audio Features")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.show()

# === STEP 4: Tempo vs Popularity Scatter Plot with Trendline ===
plt.figure(figsize=(10, 6))
sns.scatterplot(x='tempo', y='popularity', data=df_cleaned, alpha=0.6)
sns.regplot(x='tempo', y='popularity', data=df_cleaned, scatter=False, color='red', label="Trend")
plt.title("🎵 Tempo vs Popularity")
plt.xlabel("Tempo (BPM)")
plt.ylabel("Popularity Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "tempo_vs_popularity.png"))
plt.show()

# === STEP 5: Outlier Removal ===
Q1 = df_cleaned['tempo'].quantile(0.25)
Q3 = df_cleaned['tempo'].quantile(0.75)
IQR = Q3 - Q1
df_no_outliers = df_cleaned[~((df_cleaned['tempo'] < (Q1 - 1.5 * IQR)) | (df_cleaned['tempo'] > (Q3 + 1.5 * IQR)))]

# === STEP 6: Normalization ===
scaler = MinMaxScaler()
df_no_outliers.loc[:, 'energy_norm'] = scaler.fit_transform(df_no_outliers[['energy']])

# === STEP 7: Top 10 Popular Tracks Bar Chart ===
if 'track_name' in df.columns:
    top_tracks = df.sort_values('popularity', ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='popularity', y='track_name', data=top_tracks, palette="viridis")
    for i, v in enumerate(top_tracks['popularity']):
        plt.text(v + 1, i, f'{v}', va='center')
    plt.title("🔥 Top 10 Most Popular Tracks")
    plt.xlabel("Popularity Score")
    plt.ylabel("Track Name")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_10_popular_tracks.png"))
    plt.show()

# === STEP 8: Histogram of Tempo (No Outliers) ===
plt.figure(figsize=(10, 5))
sns.histplot(df_no_outliers['tempo'], kde=True, color="blue", bins=30)
plt.title("⏱️ Tempo Distribution (Outliers Removed)")
plt.xlabel("Tempo (BPM)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "tempo_distribution_no_outliers.png"))
plt.show()

# === STEP 9: Pie Chart for Musical Key Distribution ===
if 'key' in df_cleaned.columns:
    key_counts = df_cleaned['key'].value_counts().head(6)
    plt.figure(figsize=(7, 7))
    plt.pie(key_counts, labels=key_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title("🎼 Top 6 Most Common Musical Keys")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "key_distribution_pie.png"))
    plt.show()

# === STEP 10: Save CSV Outputs ===
df_cleaned.to_csv(os.path.join(output_dir, "spotify_cleaned.csv"), index=False)
df_no_outliers.to_csv(os.path.join(output_dir, "spotify_no_outliers.csv"), index=False)
