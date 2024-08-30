import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

weightage = {
    'price(₹)': 0.3,
    'manufacturer_name_embeddings': 0.1,
    'type_embeddings': 1.0,
    'primary_comp_embeddings': 0.8,
    'entire_comp_embeddings':1.0,
    'value_embeddings': 0.8,
    'entire_value_embeddings': 1.0
}

### Functions to Use

def getType(row):
    for pack_size in pack_sizes:
        if pack_size in row['pack_size_label'].lower():
            return pack_size

def getComp(row):
    return row['short_composition1'].split('(')[0].lower().rstrip()

def getValue(row):
    return row['short_composition1'].split('(')[1].lower().rstrip().rstrip(')')

def generate_word_embeddings(column, dataset):
    unique_values = dataset[column].unique()
    sentences = [[str(value)] for value in unique_values] 
    model = Word2Vec(sentences, min_count=1, vector_size=100)
    return model

# Calculate similarity scores for 'price' column using cosine similarity
def calculate_price_similarity(price1, price2):
    price1 = np.array(price1).reshape(1, -1)
    price2 = np.array(price2).reshape(1, -1)
    similarity = cosine_similarity(price1, price2)
    return similarity[0][0]

def calculate_text_similarity(value1, value2):
    similarity = cosine_similarity(value1, value2)
    return similarity[0][0]

# Calculate weighted similarity between two medicines
def calculate_weighted_similarity(medicine1, medicine2):
    similarity_scores = []
    for column, weight in weightage.items():
        if column == 'price(₹)':
            similarity = calculate_price_similarity(medicine1[column].values[0], medicine2[column])
        else:
            similarity = calculate_text_similarity(medicine1[column].values[0], medicine2[column])
        similarity_scores.append(similarity * weight)
    weighted_similarity = sum(similarity_scores)
    return weighted_similarity

try:
    ### READ Data
    df = pd.read_csv(r'medicines_dataset.csv')

    # 'type' column has only 1 value and will not have an effect on the final model
    df=df.drop(columns=['type'])
    # 'Is_discontinued','id' can also be dropped
    df = df[df.Is_discontinued==False]
    df=df.drop(columns=['Is_discontinued','id'])

    # Exploring the 'pack_size_label' column
    form=[]
    count={}
    for i in df.pack_size_label:
        words=i.split()
        x=words[-1].lower()
        if len(x)<=2:
            x=words[-2].lower()
        if x in form:
            count[x]+=1
        else:
            count[x]=1
            form.append(x)

    sorted_count = dict(sorted(count.items(), key=lambda x:x[1], reverse=True))

    # Fix: removing the plural issue (eg: tablet | tablets)
    final_count=sorted_count.copy()
    for key,value in sorted_count.items():
        test=(key+'s') in final_count
        if test:
            final_count[key]+=final_count[key+'s']
            del final_count[key+'s']
            form.remove(key+'s')
            
    final_count = dict(sorted(final_count.items(), key=lambda x:x[1], reverse=True))

    # Extract unique pack sizes from 'pack_size_label' column
    pack_sizes = df['pack_size_label'].str.extract(r'(\b\w+\b)')[0].unique()


    df['type'] = df.apply(lambda row: getType(row), axis=1)
    df['primary_comp'] = df.apply(lambda row: getComp(row), axis=1)
    df['entire_comp'] = df['primary_comp'] + ' ' + df['short_composition2'].astype(str).apply(lambda x: x.split('(')[0].lower().rstrip() if pd.notna(x) else '')
    df['value'] = df.apply(lambda row: getValue(row), axis=1)
    df['entire_value'] = df['value'] + ' ' + df['short_composition2'].astype(str).apply(lambda x: x.split('(')[1].lower().rstrip().rstrip(')') if pd.notna(x) and len(x.split('(')) > 1 else '')

    # Drop redundant columns
    dataset = df.drop(columns=['pack_size_label','short_composition1','short_composition2'])

    dataset['Disp'] = df.apply(lambda row: row['short_composition1'] if pd.isnull(row['short_composition2']) else row['short_composition1'] + ' / ' + row['short_composition2'], axis=1)

    # Define the text columns for which word embeddings need to be generated
    text_columns = ['manufacturer_name', 'type', 'primary_comp','entire_comp', 'value','entire_value']

    # Dictionary to store the models
    word_embedding_models = {}

    # Generate and store word embeddings for each text column
    for column in text_columns:
        model = generate_word_embeddings(column, dataset)
        word_embedding_models[column] = model

    # Apply word embeddings and create new columns for embeddings
    for column, model in tqdm(word_embedding_models.items(), desc="embeddings"):
        new_column_name = column + '_embeddings'
        dataset[new_column_name] = dataset[column].apply(lambda x: model.wv[str(x)].reshape(1, -1) if str(x) in model.wv else [])

    print("<<<-- SUCCESS -->>>")

except:
    print(" \(>.<)/ FAILURE - Error Occured")