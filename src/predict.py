from data_loader import load_transformations

bundle = load_transformations()
le = bundle["label_encoders"]
feature_columns = bundle["feature_columns"]

# Apply same encoding to new data
for col in bundle["column_registry"]["binary_categorical"]:
    df[col] = le[col].transform(df[col])

df = pd.get_dummies(df, columns=bundle["column_registry"]["multi_categorical"], drop_first=True)

# Align columns — handles missing dummies in new data
df = df.reindex(columns=feature_columns, fill_value=0)