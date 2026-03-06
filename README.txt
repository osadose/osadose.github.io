import pandas as pd
import pyreadstat
import numpy as np

def build_isco_kb():
    """
    Builds the ISCO knowledge base by combining the official ISCO classification
    with labeled examples from Nigeria Labour Force Survey data.
    """
    # ISCO coding scheme
    kb_isco = pd.read_excel('data/raw/ISCO.xlsx', sheet_name='ISCO_08', dtype=str)
    kb_isco['text'] = kb_isco['major_label'] + ' ' + kb_isco['sub_major_label'] + ' ' + kb_isco['minor_label'] + ' ' + kb_isco['description']
    kb_isco['text'] = kb_isco['text'].str.lower()
    kb_isco['id'] = kb_isco['unit']
    kb_isco = kb_isco[['id', 'text']]

    # Q1 2024: Main job
    kb_isco_main_q1, meta = pyreadstat.read_dta('data/raw/NLFS_2024Q1_INDIVIDUAL 1.dta', encoding='utf-8', usecols=['mjj1','mjj2a','mjj2b','mjj2cclean'])
    kb_isco_main_q1 = kb_isco_main_q1.replace(r'^\s*$', np.nan, regex=True)
    kb_isco_main_q1 = kb_isco_main_q1[kb_isco_main_q1['mjj1'].notnull()]
    kb_isco_main_q1 = kb_isco_main_q1[kb_isco_main_q1['mjj2cclean'].notnull()]
    kb_isco_main_q1['mjj2cclean'] = kb_isco_main_q1['mjj2cclean'].astype(int).astype(str)
    kb_isco_main_q1['id'] = np.where(kb_isco_main_q1.mjj2cclean.str.len() == 3, kb_isco_main_q1.mjj2cclean.str.zfill(4), kb_isco_main_q1.mjj2cclean)
    kb_isco_main_q1['text'] = kb_isco_main_q1['mjj2a'] + ' ' + kb_isco_main_q1['mjj2b']
    kb_isco_main_q1['text'] = kb_isco_main_q1['text'].str.lower()

    # Q2 2024: Main job
    kb_isco_main_q2, meta = pyreadstat.read_sav('data/raw/NLFS_2024Q2_INDIVIDUAL.sav', encoding='utf-8', usecols=['mjj1','mjj2a','mjj2b','mjj2cclean'])
    kb_isco_main_q2 = kb_isco_main_q2.replace(r'^\s*$', np.nan, regex=True)
    kb_isco_main_q2 = kb_isco_main_q2[kb_isco_main_q2['mjj1'].notnull()]
    kb_isco_main_q2 = kb_isco_main_q2[kb_isco_main_q2['mjj2cclean'].notnull()]
    kb_isco_main_q2['mjj2cclean'] = kb_isco_main_q2['mjj2cclean'].astype(int).astype(str)
    kb_isco_main_q2['id'] = np.where(kb_isco_main_q2.mjj2cclean.str.len() == 3, kb_isco_main_q2.mjj2cclean.str.zfill(4), kb_isco_main_q2.mjj2cclean)
    kb_isco_main_q2['text'] = kb_isco_main_q2['mjj2a'] + ' ' + kb_isco_main_q2['mjj2b']
    kb_isco_main_q2['text'] = kb_isco_main_q2['text'].str.lower()

    # Q1 2024: Second job
    kb_isco_second_q1, meta = pyreadstat.read_dta('data/raw/NLFS_2024Q1_INDIVIDUAL 1.dta', encoding='utf-8', usecols=['mjj1','sjj1a','sjj1b','sjj1cclean'])
    kb_isco_second_q1 = kb_isco_second_q1[kb_isco_second_q1.mjj1 == 2]
    kb_isco_second_q1 = kb_isco_second_q1[kb_isco_second_q1['sjj1cclean'].notnull()]
    kb_isco_second_q1['sjj1cclean'] = kb_isco_second_q1['sjj1cclean'].astype(int).astype(str)
    kb_isco_second_q1['id'] = np.where(kb_isco_second_q1.sjj1cclean.str.len() == 3, kb_isco_second_q1.sjj1cclean.str.zfill(4), kb_isco_second_q1.sjj1cclean)
    kb_isco_second_q1['text'] = kb_isco_second_q1['sjj1a'] + ' ' + kb_isco_second_q1['sjj1b']
    kb_isco_second_q1['text'] = kb_isco_second_q1['text'].str.lower()

    # Q2 2024: Second job
    kb_isco_second_q2, meta = pyreadstat.read_sav('data/raw/NLFS_2024Q2_INDIVIDUAL.sav', encoding='utf-8', usecols=['mjj1','sjj1a','sjj1b','sjj1cclean'])
    kb_isco_second_q2 = kb_isco_second_q2[kb_isco_second_q2.mjj1 == 2]
    kb_isco_second_q2 = kb_isco_second_q2[kb_isco_second_q2['sjj1cclean'].notnull()]
    kb_isco_second_q2['sjj1cclean'] = kb_isco_second_q2['sjj1cclean'].astype(int).astype(str)
    kb_isco_second_q2['id'] = np.where(kb_isco_second_q2.sjj1cclean.str.len() == 3, kb_isco_second_q2.sjj1cclean.str.zfill(4), kb_isco_second_q2.sjj1cclean)
    kb_isco_second_q2['text'] = kb_isco_second_q2['sjj1a'] + ' ' + kb_isco_second_q2['sjj1b']
    kb_isco_second_q2['text'] = kb_isco_second_q2['text'].str.lower()

    # Combine datasets
    kb_isco_final = pd.concat([kb_isco, kb_isco_main_q1[['id','text']], kb_isco_main_q2[['id','text']], kb_isco_second_q1[['id','text']], kb_isco_second_q2[['id','text']]])
    kb_isco_final = kb_isco_final.assign(extra_sequential_id=range(len(kb_isco_final)))
    kb_isco_final = kb_isco_final.drop_duplicates(subset=['id', 'text'], keep='first', inplace=False)
    kb_isco_final.to_csv('data/dictionaries/kb_isco.csv', columns=['extra_sequential_id','id','text'], index=False)
    print(f"ISCO knowledge base created with {len(kb_isco_final)} entries.")

def build_isic_kb():
    """
    Builds the ISIC knowledge base by combining the official ISIC classification
    with labeled examples from Nigeria Labour Force Survey data.
    """
    # ISIC coding scheme
    kb_isic = pd.read_excel('data/raw/ISIC.xlsx', sheet_name='ISIC_Rev_4', dtype=str)
    kb_isic['text'] = kb_isic['section_label'] + ' ' + kb_isic['division_label'] + ' ' + kb_isic['group_label'] + ' ' + kb_isic['description']
    kb_isic['text'] = kb_isic['text'].str.lower()
    kb_isic['id'] = kb_isic['4-digits ']
    kb_isic = kb_isic[['id', 'text']]

    # Q1 2024: Main job
    kb_isic_main_q1, meta = pyreadstat.read_dta('data/raw/NLFS_2024Q1_INDIVIDUAL 1.dta', encoding='utf-8', usecols=['mjj1','mjj3a','mjj3b','mjj3cclean'])
    kb_isic_main_q1 = kb_isic_main_q1.replace(r'^\s*$', np.nan, regex=True)
    kb_isic_main_q1 = kb_isic_main_q1[kb_isic_main_q1['mjj1'].notnull()]
    kb_isic_main_q1 = kb_isic_main_q1[kb_isic_main_q1['mjj3cclean'].notnull()]
    kb_isic_main_q1['mjj3cclean'] = kb_isic_main_q1['mjj3cclean'].astype(int).astype(str)
    kb_isic_main_q1['id'] = np.where(kb_isic_main_q1.mjj3cclean.str.len() == 3, kb_isic_main_q1.mjj3cclean.str.zfill(4), kb_isic_main_q1.mjj3cclean)
    kb_isic_main_q1['text'] = kb_isic_main_q1['mjj3a'] + ' ' + kb_isic_main_q1['mjj3b']
    kb_isic_main_q1['text'] = kb_isic_main_q1['text'].str.lower()

    # Q2 2024: Main job
    kb_isic_main_q2, meta = pyreadstat.read_sav('data/raw/NLFS_2024Q2_INDIVIDUAL.sav', encoding='utf-8', usecols=['mjj1','mjj3a','mjj3b','mjj3cclean'])
    kb_isic_main_q2 = kb_isic_main_q2.replace(r'^\s*$', np.nan, regex=True)
    kb_isic_main_q2 = kb_isic_main_q2[kb_isic_main_q2['mjj1'].notnull()]
    kb_isic_main_q2 = kb_isic_main_q2[kb_isic_main_q2['mjj3cclean'].notnull()]
    kb_isic_main_q2['mjj3cclean'] = kb_isic_main_q2['mjj3cclean'].astype(int).astype(str)
    kb_isic_main_q2['id'] = np.where(kb_isic_main_q2.mjj3cclean.str.len() == 3, kb_isic_main_q2.mjj3cclean.str.zfill(4), kb_isic_main_q2.mjj3cclean)
    kb_isic_main_q2['text'] = kb_isic_main_q2['mjj3a'] + ' ' + kb_isic_main_q2['mjj3b']
    kb_isic_main_q2['text'] = kb_isic_main_q2['text'].str.lower()

    # Q1 2024: Second job
    kb_isic_second_q1, meta = pyreadstat.read_dta('data/raw/NLFS_2024Q1_INDIVIDUAL 1.dta', encoding='utf-8', usecols=['mjj1','sjj2a','sjj2b','sjj2cclean'])
    kb_isic_second_q1 = kb_isic_second_q1.replace(r'^\s*$', np.nan, regex=True)
    kb_isic_second_q1 = kb_isic_second_q1[kb_isic_second_q1['mjj1'].notnull()]
    kb_isic_second_q1 = kb_isic_second_q1[kb_isic_second_q1['sjj2cclean'].notnull()]
    kb_isic_second_q1['sjj2cclean'] = kb_isic_second_q1['sjj2cclean'].astype(int).astype(str)
    kb_isic_second_q1['id'] = np.where(kb_isic_second_q1.sjj2cclean.str.len() == 3, kb_isic_second_q1.sjj2cclean.str.zfill(4), kb_isic_second_q1.sjj2cclean)
    kb_isic_second_q1['text'] = kb_isic_second_q1['sjj2a'] + ' ' + kb_isic_second_q1['sjj2b']
    kb_isic_second_q1['text'] = kb_isic_second_q1['text'].str.lower()

    # Q2 2024: Second job
    kb_isic_second_q2, meta = pyreadstat.read_sav('data/raw/NLFS_2024Q2_INDIVIDUAL.sav', encoding='utf-8', usecols=['mjj1','sjj2a','sjj2b','sjj2cclean'])
    kb_isic_second_q2 = kb_isic_second_q2.replace(r'^\s*$', np.nan, regex=True)
    kb_isic_second_q2 = kb_isic_second_q2[kb_isic_second_q2['mjj1'].notnull()]
    kb_isic_second_q2 = kb_isic_second_q2[kb_isic_second_q2['sjj2cclean'].notnull()]
    kb_isic_second_q2['sjj2cclean'] = kb_isic_second_q2['sjj2cclean'].astype(int).astype(str)
    kb_isic_second_q2['id'] = np.where(kb_isic_second_q2.sjj2cclean.str.len() == 3, kb_isic_second_q2.sjj2cclean.str.zfill(4), kb_isic_second_q2.sjj2cclean)
    kb_isic_second_q2['text'] = kb_isic_second_q2['sjj2a'] + ' ' + kb_isic_second_q2['sjj2b']
    kb_isic_second_q2['text'] = kb_isic_second_q2['text'].str.lower()

    # Combine datasets
    kb_isic_final = pd.concat([kb_isic, kb_isic_main_q1[['id','text']], kb_isic_main_q2[['id','text']], kb_isic_second_q1[['id','text']], kb_isic_second_q2[['id','text']]])
    kb_isic_final = kb_isic_final.assign(extra_sequential_id=range(len(kb_isic_final)))
    kb_isic_final = kb_isic_final.drop_duplicates(subset=['id', 'text'], keep='first', inplace=False)
    kb_isic_final.to_csv('data/dictionaries/kb_isic.csv', columns=['extra_sequential_id','id','text'], index=False)
    print(f"ISIC knowledge base created with {len(kb_isic_final)} entries.")

if __name__ == "__main__":
    build_isco_kb()
    build_isic_kb()








import numpy as np
import truststore
import pyarrow
import einops
import pandas as pd
import pyreadstat
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Inject truststore for secure downloads
truststore.inject_into_ssl()

# Import custom modules
from knowledgebase import build_isco_kb, build_isic_kb
from utils import compute_metrics

class MyNormalisedHF_Vectoriser:
    def __init__(self, model_name, tokenizer_kwargs=None, model_kwargs=None):
        from classifai.vectorisers import HuggingFaceVectoriser
        self.vectoriser = HuggingFaceVectoriser(
            model_name=model_name,
            tokenizer_kwargs=tokenizer_kwargs or {},
            model_kwargs=model_kwargs or {}
        )

    def transform(self, texts):
        raw_embeddings = self.vectoriser.transform(texts)
        return raw_embeddings / np.linalg.norm(raw_embeddings, axis=1, keepdims=True)

def build_vector_stores():
    """
    Builds vector stores for ISCO and ISIC using the knowledge bases.
    """
    print("Building vector stores...")
    vectoriser = MyNormalisedHF_Vectoriser(
        model_name='nomic-ai/nomic-embed-text-v1.5',
        tokenizer_kwargs={"trust_remote_code": True},
        model_kwargs={"trust_remote_code": True}
    )

    # ISCO
    from classifai.indexers import VectorStore
    vector_store_isco = VectorStore(
        file_name='data/dictionaries/kb_isco.csv',
        data_type='csv',
        vectoriser=vectoriser,
        output_dir='vector_store/isco',
        overwrite=True
    )
    print("ISCO vector store built.")

    # ISIC
    vector_store_isic = VectorStore(
        file_name='data/dictionaries/kb_isic.csv',
        data_type='csv',
        vectoriser=vectoriser,
        output_dir='vector_store/isic',
        overwrite=True
    )
    print("ISIC vector store built.")

def preprocess_input_data():
    """
    Preprocesses input data for ISCO and ISIC from raw survey data.
    """
    print("Preprocessing input data...")

    # ISCO
    query_isco = pd.read_excel('data/raw/Raw_ISICISCO_Q12025.xlsx', usecols=['jobnumber','occupationname','occupationtasksduties','isco','isco_code_clean'])
    query_isco = query_isco[~query_isco['isco_code_clean'].str.contains('l')]  # filter erroneous values
    query_isco['isco_code'] = query_isco['isco'].str.extract('(\d+)')
    query_isco = query_isco[query_isco['isco_code'].notnull()]
    query_isco = query_isco.assign(id=range(len(query_isco)))
    query_isco['full_description'] = query_isco['occupationname'] + ' ' + query_isco['occupationtasksduties']
    query_isco['full_description'] = query_isco['full_description'].str.lower()
    query_isco.to_csv('data/pre-processed/isco_2025_Q1.csv', columns=['id','full_description','isco_code','isco_code_clean'], index=False)
    print("ISCO input data preprocessed.")

    # ISIC
    query_isic = pd.read_excel('data/raw/Raw_ISICISCO_Q12025.xlsx', usecols=['jobnumber','activityname','activitygoodsservices','isic','isic_code_clean'])
    query_isic['isic_code'] = query_isic['isic'].str.extract('(\d+)')
    query_isic = query_isic[query_isic['isic_code'].notnull()]
    query_isic['isic_code_clean'] = query_isic['isic_code_clean'].astype(int).astype(str)
    query_isic['isic_code_clean'] = np.where(query_isic.isic_code_clean.str.len() == 3, query_isic.isic_code_clean.str.zfill(4), query_isic.isic_code_clean)
    query_isic = query_isic.assign(id=range(len(query_isic)))
    query_isic['full_description'] = query_isic['activityname'] + ' ' + query_isic['activitygoodsservices']
    query_isic['full_description'] = query_isic['full_description'].str.lower()
    query_isic.to_csv('data/pre-processed/isic_2025_Q1.csv', columns=['id','full_description','isic_code','isic_code_clean'], index=False)
    print("ISIC input data preprocessed.")

def search_vector_stores():
    """
    Searches the vector stores for input data and saves results.
    """
    print("Searching vector stores...")
    vectoriser = MyNormalisedHF_Vectoriser(
        model_name='nomic-ai/nomic-embed-text-v1.5',
        tokenizer_kwargs={"trust_remote_code": True},
        model_kwargs={"trust_remote_code": True}
    )

    from classifai.indexers import VectorStore
    from classifai.indexers.dataclasses import VectorStoreSearchInput

    # ISCO
    vector_store_isco = VectorStore.from_filespace('vector_store/isco', vectoriser)
    search_terms_isco = pd.read_csv('data/pre-processed/isco_2025_Q1.csv', usecols=['id', 'full_description'])
    search_input_isco = VectorStoreSearchInput({
        "id": search_terms_isco['id'].to_list(),
        "query": search_terms_isco['full_description'].to_list()
    })
    search_results_isco = vector_store_isco.search(search_input_isco, n_results=15)
    search_results_isco.to_csv('outputs/isco_2025_Q1_results.csv', index=False)
    print("ISCO search completed.")

    # ISIC
    vector_store_isic = VectorStore.from_filespace('vector_store/isic', vectoriser)
    search_terms_isic = pd.read_csv('data/pre-processed/isic_2025_Q1.csv', usecols=['id', 'full_description'])
    search_input_isic = VectorStoreSearchInput({
        "id": search_terms_isic['id'].to_list(),
        "query": search_terms_isic['full_description'].to_list()
    })
    search_results_isic = vector_store_isic.search(search_input_isic, n_results=15)
    search_results_isic.to_csv('outputs/isic_2025_Q1_results.csv', index=False)
    print("ISIC search completed.")

def evaluate_results():
    """
    Evaluates the search results against validated data and generates metrics and plots.
    """
    print("Evaluating results...")

    # ISCO Evaluation
    search_results_isco = pd.read_csv('outputs/isco_2025_Q1_results.csv')
    search_results_scores_isco = search_results_isco[['query_id','doc_id','score']].groupby(['query_id','doc_id'])['score'].max().reset_index()
    search_results_unique_isco = pd.DataFrame(search_results_isco[['query_id','doc_id']].groupby(['query_id'])['doc_id'].value_counts()).reset_index().drop('count', axis=1)
    search_results_unique_isco = search_results_unique_isco.merge(search_results_scores_isco, left_on=['query_id', 'doc_id'], right_on=['query_id', 'doc_id'])
    search_results_unique_isco['rank'] = search_results_unique_isco.groupby('query_id')['doc_id'].cumcount()
    search_results_top_isco = search_results_unique_isco[search_results_unique_isco['rank'] == 0]
    search_results_top_isco = search_results_top_isco.rename(columns={'query_id': 'id'})
    search_results_top_isco['id'] = search_results_top_isco['id'].astype(int)

    validated_isco = pd.read_csv('data/pre-processed/isco_2025_Q1.csv')
    test_results_isco = validated_isco.merge(search_results_top_isco, on='id', how='left')
    test_results_isco = test_results_isco[test_results_isco['isco_code'].notnull()]
    test_results_isco = test_results_isco[test_results_isco['doc_id'].notnull()]

    accuracy_isco = (test_results_isco['isco_code'] == test_results_isco['doc_id']).mean() * 100
    print(f'ISCO Accuracy = {round(accuracy_isco, 1)}%')

    # Threshold analysis for ISCO
    thresholds = np.arange(0, 1.01, 0.01)
    results_isco = []
    for threshold in thresholds:
        covered = test_results_isco.loc[test_results_isco['score'] > threshold]
        coverage = len(covered) / len(test_results_isco)
        accuracy = (covered['isco_code_clean'] == covered['doc_id']).mean()
        results_isco.append({'threshold': threshold, 'coverage': coverage, 'accuracy': accuracy})

    results_df_isco = pd.DataFrame(results_isco)
    print("ISCO threshold results:")
    print(results_df_isco)

    # Plot for ISCO
    x = results_df_isco['coverage']
    y = results_df_isco['accuracy']
    z = results_df_isco['threshold']
    xs = np.sort(x)
    ys = np.array(y)[np.argsort(x)]
    zs = np.array(z)[np.argsort(x)]
    x0 = 0.5
    x1 = 0.8
    y0 = np.interp(x0, xs, ys)
    y1 = np.interp(x1, xs, ys)
    z0 = np.interp(x0, xs, zs)
    z1 = np.interp(x1, xs, zs)

    sns.set_style('whitegrid', {'axes.grid': False})
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='coverage', y='accuracy', data=results_df_isco, color='#212121')
    plt.axvline(x=0.5, color='#cab2d6', ls=':', lw=2, alpha=0.8)
    plt.axvline(x=0.8, color='#6a3d9a', ls=':', lw=2, alpha=0.8)
    plt.axhline(y=y0, color='#cab2d6', ls=':', lw=2, alpha=0.8)
    plt.axhline(y=y1, color='#6a3d9a', ls=':', lw=2, alpha=0.8)
    plt.plot(x0, y0, marker='o', color='#cab2d6')
    plt.plot(x1, y1, marker='o', color='#6a3d9a')
    plt.title('Coverage vs. Accuracy for Different Similarity Thresholds (ISCO)', fontsize=14, weight='bold', y=1.055, loc='left')
    plt.gcf().text(0.125, 0.9, 'Q1 2025 NLFS - Main job (nomic-embed-text-v1.5)', fontsize=9, color='#666')
    plt.xlabel('Coverage')
    plt.ylabel('Accuracy')
    plt.text(0.02, 0.04, f'cov≈0.50> (thr={z0:.2f})\ncov≈0.80> (thr={z1:.2f})', transform=plt.gca().transAxes, fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.6'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('% 1.2f'))
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position('right')
    plt.savefig('outputs/isco_evaluation_plot.png')
    plt.show()

    # Additional accuracies for ISCO
    accuracy_pre_val = (test_results_isco['isco_code'] == test_results_isco['isco_code_clean']).mean() * 100
    print(f'In {round(accuracy_pre_val, 1)}% of cases the validated code matched the pre-validated code.')

    accuracy_pred_pre = (test_results_isco['isco_code'] == test_results_isco['doc_id']).mean() * 100
    print(f'In {round(accuracy_pred_pre, 1)}% of cases the predicted code matched the pre-validated code.')

    accuracy_pred_val = (test_results_isco['doc_id'] == test_results_isco['isco_code_clean']).mean() * 100
    print(f'In {round(accuracy_pred_val, 1)}% of cases the predicted code matched the validated code.')

    subset_isco = test_results_isco[test_results_isco['isco_code'] == test_results_isco['doc_id']]
    accuracy_agreed = (subset_isco['isco_code_clean'] == subset_isco['doc_id']).mean() * 100
    print(f'Among the rows we auto‑accept (because the model agrees with pre‑validation), how often does that agreed code match the final validated code? Answer: {round(accuracy_agreed, 1)}%')

    # ISIC Evaluation (similar structure)
    search_results_isic = pd.read_csv('outputs/isic_2025_Q1_results.csv')
    search_results_scores_isic = search_results_isic[['query_id','doc_id','score']].groupby(['query_id','doc_id'])['score'].max().reset_index()
    search_results_unique_isic = pd.DataFrame(search_results_isic[['query_id','doc_id']].groupby(['query_id'])['doc_id'].value_counts()).reset_index().drop('count', axis=1)
    search_results_unique_isic = search_results_unique_isic.merge(search_results_scores_isic, left_on=['query_id', 'doc_id'], right_on=['query_id', 'doc_id'])
    search_results_unique_isic['rank'] = search_results_unique_isic.groupby('query_id')['doc_id'].cumcount()
    search_results_top_isic = search_results_unique_isic[search_results_unique_isic['rank'] == 0]
    search_results_top_isic = search_results_top_isic.rename(columns={'query_id': 'id'})
    search_results_top_isic['id'] = search_results_top_isic['id'].astype(int)

    validated_isic = pd.read_csv('data/pre-processed/isic_2025_Q1.csv')
    test_results_isic = validated_isic.merge(search_results_top_isic, on='id', how='left')
    test_results_isic = test_results_isic[test_results_isic['isic_code'].notnull()]
    test_results_isic = test_results_isic[test_results_isic['doc_id'].notnull()]

    accuracy_isic = (test_results_isic['isic_code'] == test_results_isic['doc_id']).mean() * 100
    print(f'ISIC Accuracy = {round(accuracy_isic, 1)}%')

    # Threshold analysis for ISIC
    results_isic = []
    for threshold in thresholds:
        covered = test_results_isic.loc[test_results_isic['score'] > threshold]
        coverage = len(covered) / len(test_results_isic)
        accuracy = (covered['isic_code_clean'] == covered['doc_id']).mean()
        results_isic.append({'threshold': threshold, 'coverage': coverage, 'accuracy': accuracy})

    results_df_isic = pd.DataFrame(results_isic)
    print("ISIC threshold results:")
    print(results_df_isic)

    # Plot for ISIC (similar to ISCO)
    x = results_df_isic['coverage']
    y = results_df_isic['accuracy']
    z = results_df_isic['threshold']
    xs = np.sort(x)
    ys = np.array(y)[np.argsort(x)]
    zs = np.array(z)[np.argsort(x)]
    y0 = np.interp(x0, xs, ys)
    y1 = np.interp(x1, xs, ys)
    z0 = np.interp(x0, xs, zs)
    z1 = np.interp(x1, xs, zs)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='coverage', y='accuracy', data=results_df_isic, color='#212121')
    plt.axvline(x=0.5, color='#cab2d6', ls=':', lw=2, alpha=0.8)
    plt.axvline(x=0.8, color='#6a3d9a', ls=':', lw=2, alpha=0.8)
    plt.axhline(y=y0, color='#cab2d6', ls=':', lw=2, alpha=0.8)
    plt.axhline(y=y1, color='#6a3d9a', ls=':', lw=2, alpha=0.8)
    plt.plot(x0, y0, marker='o', color='#cab2d6')
    plt.plot(x1, y1, marker='o', color='#6a3d9a')
    plt.title('Coverage vs. Accuracy for Different Similarity Thresholds (ISIC)', fontsize=14, weight='bold', y=1.055, loc='left')
    plt.gcf().text(0.125, 0.9, 'Q1 2025 NLFS - Main job (nomic-embed-text-v1.5)', fontsize=9, color='#666')
    plt.xlabel('Coverage')
    plt.ylabel('Accuracy')
    plt.text(0.02, 0.04, f'cov≈0.50> (thr={z0:.2f})\ncov≈0.80> (thr={z1:.2f})', transform=plt.gca().transAxes, fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.6'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('% 1.2f'))
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position('right')
    plt.savefig('outputs/isic_evaluation_plot.png')
    plt.show()

    print("Evaluation completed.")

if __name__ == "__main__":
    # Build knowledge bases
    build_isco_kb()
    build_isic_kb()

    # Build vector stores
    build_vector_stores()

    # Preprocess input data
    preprocess_input_data()

    # Search vector stores
    search_vector_stores()

    # Evaluate results
    evaluate_results()
