# KNN
- 간단한 기계 학습 알고리즘
- 인스턴스 기반 학습
- Regression 과 Classification 문제에 모두 사용 가능<BR/><BR/>

:small_orange_diamond: **특징**
- 인스턴스 기반이기 때문에 내부에서 모델을 생성시키고 학습하지 않음.
- 새로운 데이터의 y값은 가장 가까운 k개의 점으로 결정됨.
- 분류는 다중 클래스로, 회귀는 k개의 점의 평균값으로 예측됨.
- 다양한 거리 척도 사용 가능
- Data Normalization 필수<BR/><BR/>

:bulb: **K-Means 알고리즘과 KNN은 다르다**
- K-Means: k개의 군집을 형성하여 클러스터링하는 비지도 학습 알고리즘
- KNN: 새로운 값을 주위 k개의 데이터로 예측하는 지도 학습 알고리즘<BR/><BR/>

:small_blue_diamond: **파이썬 코드 - 데이터 준비 단계**
  ```python
  ## 라벨 데이터
  label = pd.DataFrame(data.response)
  label.columns = ['response']
  y = label.response.values
  
  ## 특성 데이터
  feature = data.loc[:, ['x1', 'x2', 'x3', 'x4', 'x5']].copy()
  
  ## feature 데이터 표준화
  from sklearn.preprocessing import StandardScaler
  std_scaler = StandardScaler()
  feature_std = std_scaler.fit_transform(feature)
  feature_std_df = pd.DataFrame(data=feature_std, columns = list(feature.columns))
  ```
:small_blue_diamond: **파이썬 코드 - KNN 모델의 k 결정**
  ```python
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.model_selection import cross_val_score

  k_list = range(5,16,2)      ## k를 5에서 15사이 홀수로 지정
  accuracy = []
  base_score = []

  for k in k_list:
      knn = KNeighborsClassifier(n_neighbors=k, metric='mahalanobis', metric_params={'V':np.cov(feature1_std.T)})   ## k 값이 변할때마다 knn model 재정의
      accuracy.append(cross_val_score(knn, feature_std, y, scoring='f1', cv=4).mean())                              ## 각 k값에 해당하는 score 저장 - 여기서는 교차검증으로 f1 score의 평균값 저장
      base_score.append(0.8)

  ## 그래프 시각화로 결과 확인
  import matplotlib.pyplot as plt
  plt.figure(figsize=(20,8))
  plt.plot(k_list, accuracy, 'blue')
  plt.plot(k_list, base_score, 'gray')
  plt.legend(['f1 score - 4CV', 'base score'], fontsize='x-large')
  plt.ylim(0.5, 1.0)
  plt.show()

  ## score가 가장 높을때의 f1 score와 k값 확인
  k = np.array(accuracy).argmax()
  print('f1 score는 k값이 {}일때 {}'.format(2*k+5, accuracy[k]))
  
  #### 아래 그래프는 결과 그래프 예시
  ```
  <img src="https://user-images.githubusercontent.com/68538876/99638783-191b3900-2a8a-11eb-91e2-6271c35939e4.png" width="700" height="300"><BR/>  
:small_blue_diamond: **파이썬 코드 - knn 모델 구축 및 평가**
  ```python
  knn = KNeighborsClassifier(n_neighbors=15, metric='mahalanobis', metric_params={'V':np.cov(feature_std.T)})     ## 마할라노비스 거리를 이용한 KNN
  knn.fit(feature_std, y)
  y_pred = knn.predict(feature_test_std)
  print(f1_score(y_test, y_pred))
  
  ## 변수 중요도 확인
  import eli5
  from eli5.sklearn import PermutationImportance
  perm = PermutationImportance(knn, scoring='f1', random_state=0).fit(feature_std, y)
  eli5.show_weights(perm, feature_names = feature.columns.tolist())
  ```
