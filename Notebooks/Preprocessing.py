def removeWeakTornados(df_labels, df_data):
    #Removes EF0 and EF1 tornadoes
    mask_remove = df_labels.isin([0, 1])
    removed_indices = df_labels.index[mask_remove]
    df_labels = df_labels.loc[~mask_remove].reset_index(drop=True)
    df_data = df_data.drop(index=removed_indices).reset_index(drop=True)

    return df_labels, df_data