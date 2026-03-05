import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    join_cols = list(set(anon_df.columns).intersection(set(aux_df.columns)))
    join_cols = [c for c in join_cols if c not in ["anon_id", "name"]]

    # merge the two datasets
    merged = pd.merge(anon_df, aux_df, on=join_cols, how="inner")

    # count how many matches each anon_id has
    counts = merged.groupby("anon_id").size()

    # keep only anon_ids that matched exactly once
    unique_ids = counts[counts == 1].index

    unique_matches = merged[merged["anon_id"].isin(unique_ids)]

    # return required columns
    return unique_matches[["anon_id", "name"]].rename(columns={"name": "matched_name"})


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    matched = len(matches_df)
    total = len(anon_df)

    return matched / total
