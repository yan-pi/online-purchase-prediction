# Especificação do Projeto: Predição de Intenção de Compra em E-Commerce

## Objetivo

Implementar pipeline completo de Machine Learning para classificação binária de intenção de compra usando MLP e SVM, com avaliação comparativa de desempenho.

## Dataset

**UCI Online Shoppers Purchasing Intention Dataset (ID: 468)**
url: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

- Instâncias: 12.330 sessões
- Features: 17 (10 numéricas + 7 categóricas)
- Target: Revenue (binária: True/False)
- Desbalanceamento: 84,5% negativos vs 15,5% positivos
- Sem missing values
- Paper original: Sakar et al. (2019) Neural Computing & Applications

### Features

**Comportamentais (6):**

- Administrative, Informational, ProductRelated (int)
- Administrative_Duration, Informational_Duration, ProductRelated_Duration (float)

**Google Analytics (3):**

- BounceRates, ExitRates, PageValues (float)

**Temporais (3):**

- SpecialDay (float: 0-1)
- Month (categorical: Jan, Feb, Mar, Apr, May, June, Jul, Aug, Sep, Oct, Nov, Dec)
- Weekend (bool)

**Técnicas (5):**

- OperatingSystems, Browser, Region, TrafficType (int categórico)
- VisitorType (categorical: Returning_Visitor, New_Visitor, Other)

## Estrutura de Arquivos

```
projeto-ml-ecommerce/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Carregamento dataset UCI
│   ├── preprocessing.py        # Pipeline pré-processamento
│   ├── models.py               # Treinamento MLP, SVM, RF
│   └── evaluation.py           # Métricas e visualizações
├── notebooks/
│   └── 01_eda.ipynb            # Análise exploratória
├── data/                       # Dataset (auto-download)
├── results/                    # Outputs (métricas, gráficos)
├── paper/                      # Artigo LaTeX (já criado)
├── main.py                     # Pipeline principal
├── requirements.txt            # Dependências
└── README.md                   # Documentação
```

## Implementação Detalhada

### 1. data_loader.py

**Função: `load_online_shoppers()`**

```python
# Carregar dataset via ucimlrepo
from ucimlrepo import fetch_ucirepo
online_shoppers = fetch_ucirepo(id=468)
X = online_shoppers.data.features  # DataFrame 12330x17
y = online_shoppers.data.targets   # Series 12330x1
return X, y
```

**Função: `get_dataset_info(X, y)`**

- Exibir shape, tipos, estatísticas descritivas
- Contagem de classes (value_counts)
- Verificar missing values

### 2. preprocessing.py

**Classe: `OnlineShoppersPreprocessor`**

**Método: `fit_transform(X_train, y_train)`**

1. Separar features numéricas vs categóricas
2. LabelEncoder para categóricas: Month, OperatingSystems, Browser, Region, TrafficType, VisitorType
3. Converter Weekend (bool → int)
4. StandardScaler para numéricas: z = (x - μ) / σ
5. Retornar X_train_processed

**Método: `transform(X_test)`**

- Aplicar encoders e scaler já treinados
- Retornar X_test_processed

**Função: `split_data(X, y)`**

- train_test_split estratificado
- test_size=0.2
- random_state=42
- stratify=y

### 3. models.py

**Função: `train_mlp(X_train, y_train)`**

GridSearchCV com:

```python
param_grid = {
    'hidden_layer_sizes': [(100, 50), (100,), (50, 50)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['adaptive'],
    'max_iter': [500]
}
```

- MLPClassifier com class_weight='balanced'
- early_stopping=True, validation_fraction=0.1
- 5-fold CV estratificada
- scoring='f1'
- n_jobs=-1
- Retornar best*estimator*

**Função: `train_svm(X_train, y_train)`**

GridSearchCV com:

```python
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']
}
```

- SVC com class_weight='balanced'
- probability=True (para AUC-ROC)
- 5-fold CV estratificada
- scoring='f1'
- n_jobs=-1
- Retornar best*estimator*

**Função: `train_random_forest(X_train, y_train)`**

GridSearchCV com:

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}
```

- RandomForestClassifier
- 5-fold CV estratificada
- scoring='f1'
- n_jobs=-1
- Retornar best*estimator*

### 4. evaluation.py

**Função: `evaluate_model(model, X_test, y_test, model_name)`**

Calcular e retornar dict com:

```python
{
    'Model': model_name,
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred),
    'AUC-ROC': roc_auc_score(y_test, y_proba)
}
```

**Função: `plot_confusion_matrix(y_test, y_pred, model_name, save_path)`**

- Gerar matriz de confusão com seaborn.heatmap
- Anotar valores (TN, FP, FN, TP)
- Salvar PNG em results/

**Função: `plot_roc_curve(models_dict, X_test, y_test, save_path)`**

- models_dict: {'MLP': model_mlp, 'SVM': model_svm, ...}
- Plotar curvas ROC para todos modelos no mesmo gráfico
- Incluir linha diagonal (random classifier)
- Legendas com AUC de cada modelo
- Salvar PNG em results/

**Função: `compare_models(results_list)`**

- results_list: lista de dicts de evaluate_model()
- Retornar DataFrame comparativo
- Salvar CSV em results/model_comparison.csv

**Função: `print_classification_report(y_test, y_pred, model_name)`**

- classification_report do sklearn
- Exibir no console

### 5. main.py

Pipeline principal:

```python
def main():
    # 1. Carregar dados
    X, y = load_online_shoppers()
    get_dataset_info(X, y)

    # 2. Split estratificado
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Pré-processamento
    preprocessor = OnlineShoppersPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_test_proc = preprocessor.transform(X_test)

    # 4. Treinamento
    print("Treinando MLP...")
    model_mlp = train_mlp(X_train_proc, y_train)

    print("Treinando SVM...")
    model_svm = train_svm(X_train_proc, y_train)

    print("Treinando Random Forest...")
    model_rf = train_random_forest(X_train_proc, y_train)

    # 5. Avaliação
    results = []
    results.append(evaluate_model(model_mlp, X_test_proc, y_test, 'MLP'))
    results.append(evaluate_model(model_svm, X_test_proc, y_test, 'SVM'))
    results.append(evaluate_model(model_rf, X_test_proc, y_test, 'Random Forest'))

    # 6. Visualizações
    plot_confusion_matrix(y_test, model_mlp.predict(X_test_proc),
                         'MLP', 'results/confusion_matrix_mlp.png')
    plot_confusion_matrix(y_test, model_svm.predict(X_test_proc),
                         'SVM', 'results/confusion_matrix_svm.png')
    plot_confusion_matrix(y_test, model_rf.predict(X_test_proc),
                         'Random Forest', 'results/confusion_matrix_random_forest.png')

    models = {'MLP': model_mlp, 'SVM': model_svm, 'Random Forest': model_rf}
    plot_roc_curve(models, X_test_proc, y_test, 'results/roc_curves.png')

    # 7. Comparação final
    df_results = compare_models(results)
    print("\n=== RESULTADOS FINAIS ===")
    print(df_results.to_string(index=False))

if __name__ == '__main__':
    main()
```

### 6. notebooks/01_eda.ipynb

Análise exploratória com células:

1. **Setup e carregamento**

```python
from src.data_loader import load_online_shoppers, get_dataset_info
X, y = load_online_shoppers()
```

2. **Informações básicas**

- Shape, tipos, describe()
- Contagem de classes (bar + pie chart)

3. **Features numéricas**

- Histogramas (6 principais)
- Matriz de correlação (heatmap)
- Box plots por classe

4. **Features categóricas**

- Value counts e bar plots
- Distribuição por mês
- VisitorType breakdown

5. **Relações bivariadas**

- PageValues vs Revenue (box plot)
- BounceRates vs ExitRates (scatter)
- ProductRelated vs Revenue

6. **Verificações**

- Missing values: `X.isnull().sum()`
- Outliers: box plots
- Desbalanceamento: proporções

### 7. requirements.txt

```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
ucimlrepo==0.0.7
imbalanced-learn==0.11.0
jupyter==1.0.0
notebook==7.0.2
```

### 8. README.md

Seções:

- Título e descrição
- Dataset (link UCI, características)
- Instalação (3 comandos)
- Execução (python main.py)
- Estrutura de arquivos
- Modelos implementados (MLP, SVM, RF)
- Métricas (5 principais)
- Resultados (placeholder para preencher)
- Compilação LaTeX (comandos pdflatex + bibtex)
- Licença

## Requisitos Técnicos

### Pré-processamento

- ✓ Label encoding para categóricas nominais
- ✓ Conversão booleana → int
- ✓ StandardScaler para numéricas (fit em treino, transform em teste)
- ✓ Split estratificado 80/20

### Modelos

- ✓ MLP com GridSearchCV (topologias, alpha, early_stopping)
- ✓ SVM com GridSearchCV (kernels RBF/linear, C, gamma)
- ✓ Random Forest como baseline
- ✓ class_weight='balanced' em todos
- ✓ 5-fold CV estratificada
- ✓ scoring='f1'

### Avaliação

- ✓ Accuracy, Precision, Recall, F1-Score, AUC-ROC
- ✓ Matrizes de confusão (3 PNGs)
- ✓ Curvas ROC comparativas (1 PNG)
- ✓ DataFrame comparativo (CSV)
- ✓ Classification reports no console

### Outputs Esperados

```
results/
├── model_comparison.csv
├── confusion_matrix_mlp.png
├── confusion_matrix_svm.png
├── confusion_matrix_random_forest.png
└── roc_curves.png
```

## Métricas de Sucesso

**Mínimo esperado:**

- F1-Score: 0,60-0,75 (dado desbalanceamento)
- AUC-ROC: 0,85-0,92
- Recall: 0,55-0,70 (capturar compradores)
- Precision: 0,65-0,80 (evitar falsos positivos)

**Validações:**

- MLP deve convergir (< 500 épocas)
- SVM RBF geralmente supera linear neste dataset
- RF serve como sanity check (baseline sólido)
- class_weight='balanced' melhora recall significativamente vs default

## Próximos Passos (Pós-Implementação)

1. Executar pipeline: `python main.py`
2. Verificar saídas em `results/`
3. Copiar métricas de `model_comparison.csv` para Tabela 1 do artigo LaTeX
4. Compilar artigo: `cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
5. Preparar apresentação com:
   - Introdução ao problema (2 min)
   - Dataset e pré-processamento (2 min)
   - Modelos e hiperparâmetros (3 min)
   - Resultados (gráficos + tabela, 3 min)
   - Conclusões e trabalhos futuros (2 min)

## Notas de Implementação

- Usar `random_state=42` em todas operações estocásticas
- Validar tipos de dados após encoding (não aceitar object em X_processed)
- GridSearchCV pode demorar 5-15 minutos (total ~30 min)
- Salvar prints de métricas no console para relatório
- Matplotlib backend: 'Agg' se rodar em servidor sem display
- Verificar convergência MLP: se não convergir em 500 épocas, aumentar max_iter
- SVM com kernel RBF pode ser lento: considerar cache_size=500 se necessário

## Decisões de Design

**Por que MLP vs SVM?**

- MLP: aprende features hierárquicas, flexível
- SVM: margin-based, kernel trick, teoricamente fundamentado
- Ambos adequados para binary classification com tabular data

**Por que class_weight='balanced'?**

- Dataset 84,5% / 15,5% requer ponderação
- Alternativa seria SMOTE (não usado para manter simplicidade)

**Por que F1 como métrica de otimização?**

- Balanceia precision e recall
- Mais apropriado que accuracy para desbalanceamento
- Evita modelos que classificam tudo como negativo

**Por que StandardScaler?**

- SVM sensível a escalas (distâncias euclidianas)
- MLP converge mais rápido com normalização
- Preserva relações lineares (vs MinMaxScaler)

**Por que GridSearchCV vs RandomizedSearchCV?**

- Grid pequeno (< 100 combinações por modelo)
- Preferível busca exaustiva para reprodutibilidade
- RandomizedSearch útil para grids > 1000 combinações
