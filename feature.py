import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Read both CSV files
clean_df = pd.read_csv('original_features.csv')
stego_df = pd.read_csv('stego_features.csv')

# 2. Filter out non-numeric columns (like filenames) right away from both
# (Assuming your features are numeric and any image name/path is a string)
X_clean = clean_df.select_dtypes(include=['number'])
X_stego = stego_df.select_dtypes(include=['number'])

# 3. Create explicit target labels for each dataset
# 0 = Clean/Cover, 1 = Stego
y_clean = np.zeros(X_clean.shape[0])
y_stego = np.ones(X_stego.shape[0])

# 4. Combine (Concatenate) the datasets vertically
X_combined = pd.concat([X_clean, X_stego], axis=0, ignore_index=True)
y_combined = np.concatenate([y_clean, y_stego])

# 5. Normalize Features across the entire combined dataset
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_combined)

# 6. Train/Test Split
# We add `stratify=y_combined` to ensure an equal balance of clean/stego images in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_combined, test_size=0.2, random_state=42, stratify=y_combined
)

# Verify the shapes of the splits
print("Data pipeline completed successfully!")
print(f"Total dataset size: {X_combined.shape[0]} images")
print(f"X_train shape: {X_train.shape} | Features trained on.")
print(f"y_train distribution: Clean={np.sum(y_train==0)}, Stego={np.sum(y_train==1)}")
print(f"X_test shape:  {X_test.shape}  | Features held out for testing.")
print(f"y_test distribution:  Clean={np.sum(y_test==0)}, Stego={np.sum(y_test==1)}")