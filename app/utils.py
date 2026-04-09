import pandas as pd

def build_applicant_from_dict(d, expected_cols):
    df = pd.DataFrame([d])

    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].str.strip() #column values

    # ensure column exist
    missing_columns = [c for c in expected_cols if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing input columns: {missing_columns}")

    return df[expected_cols]
