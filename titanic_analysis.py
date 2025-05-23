import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Aula 6 — Projeto prático: Análise de sobrevivência no Titanic *************************
df1 = pd.read_csv("./dataset/train_fake.csv")
df2 = pd.read_csv("./dataset/titanic-data-6.csv")

print("Colunas do train_fake.csv:")
print(df1.columns)

print("\nColunas do titanic-data-6.csv:")
print(df2.columns)

print("\nPrimeiras linhas de train_fake.csv:")
print(df1.head())

print("\nPrimeiras linhas de titanic-data-6.csv:")
print(df2.head())

# 1. Quantidade total de passageiros
print(f"\nTotal de passageiros: {len(df1)}")

# 2. Taxa de sobrevivência geral
sobreviveu = df1["Survived"].sum()
total = len(df1)
print(f"Taxa de sobrevivência: {sobreviveu / total:.2%}")

# 3. Distribuição por sexo
print(df1["Sex"].value_counts())

# 4. Taxa de sobrevivência por sexo
print(df1.groupby("Sex")["Survived"].mean())

# 5. Distribuição por classe
print(df1["Pclass"].value_counts())

# 6. Taxa de sobrevivência por classe
print(df1.groupby("Pclass")["Survived"].mean())

# Aula 7 — Visualizações Cruzadas com Seaborn *******************************************

# 1. Sobrevivência por sexo
sns.countplot(data=df1, x="Sex", hue="Survived")
plt.title("Sobrevivência por Sexo")
plt.xlabel("Sexo")
plt.ylabel("Quantidade")
plt.legend(title="Sobreviveu", labels=["Não", "Sim"])
plt.show()

# 2. Sobrevivência por classe
sns.countplot(data=df1, x="Pclass", hue="Survived")
plt.title("Sobrevivência por Classe")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.legend(title="Sobreviveu", labels=["Não", "Sim"])
plt.show()

# 3. Distribuição de idade
sns.histplot(data=df1, x="Age", bins=30, kde=True)
plt.title("Distribuição de Idade")
plt.xlabel("Idade")
plt.ylabel("Quantidade")
plt.show()

# 4. Sobrevivência por idade
sns.boxplot(data=df1, x="Survived", y="Age")
plt.title("Idade vs Sobrevivência")
plt.xlabel("Sobreviveu (0 = não, 1 = Sim)")
plt.ylabel("Idade")
plt.show()

# 5. Mapa de calor para variáveis numéricas
# Pega apenas colunas numéricas para correlação
df_numerico = df1.select_dtypes(include=["float64", "int64"])

# Agora sim: mapa de calor da correlação
sns.heatmap(df_numerico.corr(), annot=True, cmap="YlGnBu")
plt.title("Correlação entre Variáveis Numéricas")
plt.show()
print("\n")

# Próximos Passos: Preparação para Machine Learning *************************************

# 1. Selecionar colunas relevantes
colunas_uteis = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

df_modelo = df1[colunas_uteis]

# 2. Tratar valores ausentes
print(df_modelo.isnull().sum())

df_modelo.loc[:, 'Age'] = df_modelo['Age'].fillna(df_modelo['Age'].median())
df_modelo.loc[:, 'Embarked'] = df_modelo['Embarked'].fillna(df_modelo['Embarked'].mode()[0])


# 3. Transformar variáveis categóricas em números
df_modelo = pd.get_dummies(df_modelo, columns=["Sex", 'Embarked'], drop_first=True)

# 4. Separar X (features) e y (alvo)
X = df_modelo.drop("Survived", axis=1)
y = df_modelo["Survived"]

# 5. Dividir em treino e teste
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Treinar um modelo simples
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)

# Previsões
y_pred = modelo.predict(X_test)

# Acurácia
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")

# 1. Métricas adicionais:
from sklearn.metrics import classification_report
print(classification_report(y_test, modelo.predict(X_test)))

# 2. Matriz de confusão:
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, modelo.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

# 3. Exportar o modelo treinado:
import joblib
joblib.dump(modelo, 'modelo_titanic.pkl')
