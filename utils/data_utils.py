import pandas as pd
import numpy as np

def generate_code_list(drugtype, concept_class):
    sql_query = ("SELECT ancestor_concept_id, descendant_concept_id, concept_name " + 
               "FROM dbo.concept_ancestor JOIN dbo.concept ON descendant_concept_id = concept_id "+
               "WHERE ancestor_concept_id = (SELECT concept_id from dbo.concept WHERE concept_class_id = '"+concept_class+"' AND concept_name = '"+drugtype+"');")
    codes_list = pd.read_sql(sql_query, conn)
    return list(codes_list['descendant_concept_id'])

def drop_rare_occurrences(df, col_concept, col_id, size_pop):
    unique_occurrences = df[['person_id', col_concept]].drop_duplicates()
    unique_occurrences = unique_occurrences.value_counts(col_concept)
    common_occurrences = unique_occurrences[unique_occurrences/size_pop > 0.01].index
    return df.loc[df[col_concept].isin(common_occurrences)]

def drop_unshared_features(df, col_concept, list_cols):
    print(len(df))
    df = df.loc[df[col_concept].isin(list_cols)]
    print(len(df))
    return df

"""
Pad the data: add 0s to the beginning of each patient trajectory
- Get the max timestep for each patient --> subtract from the max possible timestep
- Add this max per patient to each iteration so that we now have "backwards-aligned" patient timesteps
- Pad the earliest timesteps with 0
"""

def get_full_df(original_df, scaled_df_mat, pids, overall_max):
    # get the maximum iterations per patient and subtract from the number of timesteps in the matrix 
    # for psychosis SCZ, that is 41
    save_cols = original_df.columns
    original_df = original_df[original_df.columns[0:1]]
    max_iter = original_df.reset_index().groupby('person_id')['ranked_iteration'].max()
    max_iter.name = 'max_iter'
    max_iter = overall_max-max_iter
    
    # add the number of padding rows that need to happen per patient to the dataframe
    original_df = original_df.merge(max_iter, how='left', left_index=True, right_index=True)
    original_df.reset_index(inplace=True)
    original_df['ranked_iteration'] = original_df['ranked_iteration']+original_df['max_iter']
    
    # replace the data with the scaled data
    original_df.set_index(['person_id', 'ranked_iteration'], inplace=True)
    original_df.drop('max_iter', axis=1, inplace=True)
    
    # create a new dataframe that goes through each patient and each timestep
    new_df = pd.DataFrame(index=[np.repeat(pids, overall_max), np.tile(np.arange(1, overall_max+1), len(pids))], columns=save_cols)
    
    # then fill it in with the existing data
    new_df.loc[original_df.index] = scaled_df_mat
    
    # convert to matrix and fillna
    new_df = new_df.values.astype(float)
    new_df[np.isnan(new_df)] = 0
    return new_df

