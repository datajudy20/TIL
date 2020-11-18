# 데이터 전처리 Python 코드 (Data Preprocessing)
###### 동국대학교 데이터사이언스 소프트웨어 `데이터사이언스' 전공 강의에서 공부하며 정리한 내용 정리
---
### 데이터 불러오기
- csv 파일  
```data = pd.read_csv('directory.csv', na_values=[])```
- xlsx 파일  
```data = pd.read_excel('directory.xlsx', na_values = [], sheet_name= )```  
### 누락된 데이터 다루기
- 결측값 확인 및 삭제  
  ```Python
  df.isnull().sum()           # 열마다 결측값 개수 출력 
  df.dropna(axis=0)           # 결측값이 있는 행 삭제
  df.dropna(axis=1)           # 결측값이 있는 열 삭제
  df.dropna(how='all')        # 모든 열이 결측일 때만 행 삭제
  df.dropna(subset=['C'])     # C열에 결측이 있는 행 삭제
  #### 결측값이 삭제된 데이터프레임을 저장하고 싶을땐, inplace=True 옵션 사용
  ```  
- 결측값 대체  
  ```Python
  ## 방법 1 ##
  df.fillna(0)                # 0으로 결측값 대체
  df.fillna(df.mean())        # 각 열의 평균으로 결측값 대체
  #### 결측값이 대체된 데이터프레임을 저장하고 싶을땐, inplace=True 옵션 사용
  ```
  ```Python
  ## 방법 2 ##
  from sklearn.preprocessing import Imputer
  imr = Imputer(missing_values='NaN', strategy='mean', axis=0)    # 각 열의 평균값으로 NaN으로 표기된 결측값 대체 (최빈값도 가능)
  imputed_data = imr.fit_transform(df.values)                     
  ```
### 범주형 데이터 다루기
- 순서가 있는 특성 수치화
  ```Python
  size_mapping = {'XL' : 3, 'L' : 2, 'M' : 1}
  df['size'] = df['size'].map(size_mapping)
  ```
- 순서가 없는 특성 라벨 인코딩
  ```Python
  from sklearn.preprocessing import LabelEncoder
  encoder = LabelEncoder()
  df['label'] = encoder.fit_transform(df['label'])
  ```
- 순서가 없는 특성 원-핫 인코딩
  ```Python
  ## 방법 1 ##
  df = pd.get_dummies(df, columns=['color'], drop_first=True)
  ```
  ```Python
  ## 방법 2 ##
  from sklearn.preprocessing import OneHotEncoder
  encoder = OneHotEncoder(drop='first')
  df['color'] = encoder.fit_transform(df['color'])
  ```
### 상관관계 파악
- heatmap 그래프
  ```Python
  import matplotlib.pyplot as plt
  import seaborn as sns
  plt.figure(figsize=(15,15))
  sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='RdYlGn_r')
  plt.show()
  ```
- pairplot 그래프
  ```Python
  import matplotlib.pyplot as plt
  import seaborn as sns
  plt.figure(figsize=(15,15))
  sns.pairplot(data, hue=y, markers=['o', 's'])
  plt.show()
  ```
### 스케일 맞추기
- 정규화 (최소-최대 스케일 변환)<BR/>: 데이터를 특정 구간으로 바꾸는 척도법으로 범위가 정해진 값이 필요할 때 사용함
  ```Python
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  X_train_norm = scaler.fit_transform(X_train)    # train set으로 학습한 후 train set 변환
  X_test_norm = scaler.transform(X_test)          # train set으로 학습된 parameter로 test set 변환
  ```
- 표준화<BR/>: 평균 0, 표준편차를 1로 만들어 정규분포와 같은 특징을 갖도록 한다. 가중치를 더 쉽게 학습할 수 있도록 하고 이상치에 덜 민감하다.
  ```Python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_train_std = scaler.fit_transform(X_train)    # train set으로 fit한 후 변환
  X_test_std = scaler.transform(X_test)          # test set 변환
  ```
### 차원 축소
- PCA<BR/>: p개의 feature를 선형결합해서 새로운 feature 생성함. 큰 variance를 보이는 feature를 많은 정보량을 지닌 변수로 보고 feature 간의 관계는 covariance로 확인함. 직교하는 특성 축을 따라 분산이 최대가 되는 저차원 부분 공간으로 데이터를 투영함.
  ```Python
  from sklearn.decomposition import PCA
  pca = PCA(n_components=2)                     # 2차원으로 축소
  X_train_pca = pca.fit_transform(X_train_std)  # train set으로 fit한 후 변환
  np.cumsum(pca.explained_variance_ratio_)      # 2개의 pca 변수의 누적 설명 비율
  pca.components_                               # 2개의 pca 변수 수식 - 선형결합에서 각 변수 계수
  X_test_pca = pca.transform(X_test_std)        # test set 변환
  ```
### 군집화
- K-Means Clustering<BR/>: 거리를 기준으로 관측치들을 미리 지정된 수의 군집으로 나눔
  ```Python
  from sklearn.cluster import KMeans
  kmean = KMeans(n_clusters=2, random_state=0)
  kmean.fit(X)
  kmean.labels_
  kmean.cluster_centers_
  pd.Series(kmeans.labels_).value_counts()
  pred = kmean.predict(X_new)
  ```
- Hierarchical Clustering<BR/>: 거리를 기준으로 각 관측치가 하나씩 연결되어 군집으로 묶임
  ```Python
  from sklearn.cluster import AgglomerativeClustering
  hclust = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')    #linkage 옵션 : 완전연결, 평균연결, 단일연결, 무게중심연결
  hclust.fit_predict(X)
  hclust.labels_
  ```
