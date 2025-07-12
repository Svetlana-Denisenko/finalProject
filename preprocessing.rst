.. code:: ipython3

    !pip install pymorphy3
    !pip install nltk
    !pip install gensim
    !pip install openpyxl
    !pip install torch transformers

.. code:: ipython3

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, Dataset
    from torch.utils.data.dataloader import default_collate
    import torch.nn.functional as F
    
    
    import gensim.downloader as api
    from gensim.models import Word2Vec
    
    import matplotlib.pyplot as plt
    import nltk
    from nltk.corpus import stopwords
    from nltk.util import ngrams
    from nltk.tokenize import word_tokenize
    
    from itertools import chain
    import seaborn as sns
    
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows
    
    import numpy as np
    import pandas as pd
    import pickle
    import re
    import seaborn as sns
    import time
    import os
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import LabelEncoder
    from sklearn.decomposition import TruncatedSVD
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import SpectralClustering
    from sklearn.manifold import TSNE
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    from scipy.sparse import save_npz, load_npz
    from scipy.sparse import vstack
    from scipy.sparse import csr_matrix
    
    from collections import Counter, defaultdict
    
    from pymorphy3 import MorphAnalyzer
    
    from tqdm.auto import tqdm, trange

.. code:: ipython3

    # Загрузка списка стоп-слов из NLTK
    nltk.download('stopwords')
    stop_words_nltk = set(stopwords.words('russian'))  # Используем русский список стоп-слов
    
    # Лемматизатор
    morph = MorphAnalyzer()
    lem_cache = defaultdict(list)  # Кеширование результатов лемматизации
    
    def preprocess_data(file_path, batch_size=1000):
    
        print("Чтение файла...")
        df = pd.read_excel(file_path)
    
        # Стандартизация типов данных
        df['VOICE'] = df['VOICE'].astype(str)
        df['STEP_NAME'] = df['STEP_NAME'].astype(str)
    
        # Маска для фильтрации нужных записей
        mask_supervised = ~df['STEP_NAME'].isin(['Не распознано', 'Уточняем','Оператор'])
        df_supervised = df.loc[mask_supervised].copy()
    
        # Предварительная очистка колонки VOICE
        print("Предварительная обработка данных...")
        df_supervised['VOICE'] = (
            df_supervised['VOICE']
                .str.split('|')
                .str[0]
                .str.replace(r'[^\w\s]', '', regex=True)
        )
    
        # Удаление стоп-слов с помощью NLTK
        df_supervised['VOICE'] = df_supervised['VOICE'].apply(
            lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words_nltk])
        )
    
        # Убираем строки с пустой колонкой VOICE
        df_supervised = df_supervised[df_supervised['VOICE'].str.strip().ne('')]
    
        # Процесс лемматизации
        print("Начало лемматизации...")
    
        # Функциональность лемматизации с кешированием
        def lemmatize_batch(batch):
            result = []
            for text in batch:
                words = text.split()
                lemmas = []
                for word in words:
                    if word not in lem_cache:
                        lem_cache[word] = morph.parse(word)[0].normal_form
                    lemmas.append(lem_cache[word])
                result.append(' '.join(lemmas))
            return result
    
        # Разделение данных на батчи и постепенная обработка
        total_batches = int(len(df_supervised) / batch_size) + 1
        results = []
        with tqdm(total=total_batches, unit="%", bar_format='{l_bar}{bar}| {percentage:.1f}%') as pbar:
            for i in range(0, len(df_supervised), batch_size):
                start_idx = i
                end_idx = min(i + batch_size, len(df_supervised))
    
                # Берём партию данных
                batch = df_supervised.iloc[start_idx:end_idx]['VOICE']
    
                # Применяем лемматизацию к партии
                processed_batch = lemmatize_batch(batch)
    
                # Сохраняем обработанные данные
                results.extend(processed_batch)
    
                # Обновляем прогресс-бар
                pbar.update(1)
    
        # Создаем новый столбец VOICE_LEM и присваиваем туда лемматизированные данные
        df_supervised['VOICE_LEM'] = results
    
        return df_supervised
    
    if __name__ == "__main__":
        # Основной путь к данным для моделирования
        file_path_model = '/content/lsir_recognition_2025-06-30.xlsx'
    
        # Обработка основного набора данных
        supervised_df = preprocess_data(file_path_model)
    
        # Вывод первых записей каждой таблицы
        print(supervised_df)


.. parsed-literal::

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.
    

.. parsed-literal::

    Чтение файла...
    Предварительная обработка данных...
    Начало лемматизации...
    


.. parsed-literal::

      0%|          | 0.0%


.. parsed-literal::

                                STEP_NAME  \
    0                      Оформить карту   
    2                      МБ/FIX/SPUTNIK   
    5       Статус заявки на кредит/карту   
    9                   Справки и выписки   
    13                     МБ/FIX/SPUTNIK   
    ...                               ...   
    246583             Ежемесячный платеж   
    246584        Блокировка или закрытие   
    246590       Изменить кредитный лимит   
    246592                  Остаток (ПДП)   
    246599                 МБ/FIX/SPUTNIK   
    
                                                        VOICE  \
    0       оформил карту кредитной улучшение кредитной ис...   
    2                          соедини установкой телевидения   
    5                          хотел заказать кредитную карту   
    9                                 запросить выписку конец   
    13                                       закрытие кредита   
    ...                                                   ...   
    246583                                    проходит платеж   
    246584                                              верно   
    246590                                     изменить лимит   
    246592                                    остаток кредиту   
    246599                                     поменять тариф   
    
                                                    VOICE_LEM  
    0       оформить карта кредитный улучшение кредитный и...  
    2                         соединить установка телевидение  
    5                         хотеть заказать кредитный карта  
    9                                 запросить выписка конец  
    13                                        закрытие кредит  
    ...                                                   ...  
    246583                                   проходить платёж  
    246584                                              верно  
    246590                                     изменить лимит  
    246592                                     остаток кредит  
    246599                                     поменять тариф  
    
    [50353 rows x 3 columns]
    
