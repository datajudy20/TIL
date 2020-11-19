# XGBoost

- Gradient Boosting 알고리즘을 분산 환경에서 실행할 수 있도록 구현한 라이브러리.
- 여러 개의 Decision Tree를 조합해서 사용하는 앙상블 알고리즘.
- Regression 과 Classification 문제 모두 지원함<BR/><BR/>

:small_orange_diamond: **특징**
- gbm 보다 빠름
- 과적합 방지가 가능한 규제가 포함되어 있음
- CART (Classification And Regression Tree)를 기반으로 함
- Gradient Boost를 기반으로 함 (앙상블 부스팅의 특징인 가중치 부여를 경사하강법으로 진행함)
- greedy-algorithm을 사용한 자동 가지치기가 가능함 ⇒ 과적합이 잘 일어나지 않음
- 조기 종료를 제공함. 오류가 더 이상 개선되지 않으면 수행을 중지함.<BR/> n_estimators를 200으로 설정하고 조기 중단 파라미터 값을 50으로 설정하면, 1부터 200회까지 부스팅을 반복하다가 50회를 반복하는 동안 학습 오류가 감소하지 않으면 더 이상 부스팅을 진행하지 않고 종료함.
<BR/>(100회에서 학습 오류 값이 0.8인데 101~150회 반복하는 동안 예측 오류가 0.8보다 작은 값이 하나도 없으면 부스팅 종료)<BR/><BR/>

:small_orange_diamond: **파라미터**
- **booster** : 어떤 부스터 구조를 쓸지 결정하는 파라미터. (gbtree / gblinear / dart)
- nthread : 몇 개의 thread를 동시에 처리할지 결정하는 파라미터. 병렬처리에 사용되는 코어수.
- num_feature : feature 차원의 숫자를 결정하는 파라미터.
- **learning rate** : 학습 단계별로 가중치를 얼마나 적용할 지 결정하는 파라미터. 과적합 문제를 방지하려고 사용함. 일반적으로 0.01 ~ 0.2 값이 사용됨. (default=0.3)
- **n_estimators** : 생성할 weak learner 개수. learning rate 값을 낮추면 이 값을 반대로 높여주어야함.
- min_child_weight : 과적합을 방지하는 목적으로 사용되는 파라미터. 높은 값은 과소적합을 야기함.
- min split loss : information gain에 페널티를 부여하는 파라미터. 값이 커질수록 의사결정나무들은 가지를 잘 만들려 하지 않아서 트리 깊이가 줄어듬. (default=0)
- **max depth** : 의사결정나무의 최대 깊이. 값이 커질수록 더 복잡한 모델이 생성되며 과적합 문제를 일으킬 수 있음. 일반적으로 3 ~ 10 값이 사용됨. (default=6)
- reg_lambda : L2 Regularization Form에 달리는 weights 파라미터. 숫자가 클수록 보수적인 모델이 됨.
- **reg_alpha** : L1 Regularization Form weights 파라미터. 숫자가 클수록 보수적인 모델이 됨. 차원이 높은 경우 알고리즘 속도를 높일 수 있음.
- **objective** : 목적함수를 정의하는 파라미터.<BR/>
reg:linear - 회귀 모델<BR/>
binary:logistic - 이항 분류 로지스틱 회귀 모형. 예측 확률을 반환함<BR/>
multi:softmax - 다항 분류 문제. 클래스를 반환함. num_class를 지정해야함.<BR/>
multi:softprob - 다항 분류 문제. 각 클래스 범주에 속하는 예측 확률을 반환함
- eval_metric : 모델의 평가 함수를 조정하는 파라미터. 설정한 objective별로 기본 설정값이 지정되어 있음.<BR/>(rmse / mae / logloss / error / merror / mlogloss / map / auc 등)
- num_rounds : 부스팅 라운드를 결정하는 파라미터. 랜덤하게 생성되는 모델이므로 값이 적당히 큰게 좋다. epoch 옵션과 동일함.
- **subsample** : 훈련 데이터에서 subset을 만들지 전부를 사용할지 정하는 파라미터. 매번 나무를 만들 때 적용하며 과적합 문제를 방지하려고 사용함. 일반적으로 0.5 ~ 1 사용됨.(default=1)
- **colsample_bytree** : 나무를 만들 때 변수를 샘플링해서 쓸지 결정하는 파라미터. 나무를 만들기 전 한번 샘플링을 하게 됨. 일반적으로 0.5 ~ 1 사용됨. (default=1)
- scale_pos_weight : 분류 모델에서 사용하는 가중치 파라미터. 클래스 불균형이 심한 경우 0보다 큰 값을 지정하면 유용함. (default=1)

              

##### :bulb: 모델의 하이퍼 파라미터는 대부분 값이 정해진 규칙이 없어서 주로 랜덤서치, 그리드서치, 또는 베이지안 최적화를 이용한 방법으로 최적값을 찾음.<BR/><BR/><BR/>


:small_blue_diamond: **파이썬 코드 - 파라미터 최적값 찾기 (베이즈 최적화 방법)**
```python
import xgboost as xgb
xgb_clf = xgb.XGBClassifier()

### 베이즈 최적화 ###
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization

def XGB_cv(max_depth, n_estimators, min_child_weight, subsample, colsample_bytree, learning_rate, silent=True, nthread=-1):
  model = xgb.XGBClassifier(max_depth=int(max_depth),
                            n_estimators=int(n_estimators),
                            min_child_weight=min_child_weight,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            learning_rate=learning_rate,
                            silent=silent,
                            nthread=nthread)
  F1 = cross_val_score(model, X, y, scoring='f1', cv=3).mean()
  return F1
  
# 주어진 범위 사이에서 적절한 값을 찾는다.
pbounds = {'max_depth': (4, 7),
          'learning_rate': (0.01, 0.1),
          'n_estimators': (500, 1000),
          'min_child_weight': (3, 9),
          'subsample': (0.7, 0.99),
          'colsample_bytree' :(0.7, 0.99)
          }

xgboostBO = BayesianOptimization(f = XGB_cv, pbounds = pbounds, verbose = 2, random_state = 0)
xgboostBO.maximize(init_points=10, n_iter = 30)
xgboostBO.max                                       ## 찾은 파라미터 값 확인
```

:small_blue_diamond: **파이썬 코드 - 모델 학습 및 평가**

```python
xgb_model = xgb.XGBClassifier(max_depth= int(xgboostBO.max['params']['max_depth']),
                             learning_rate=xgboostBO.max['params']['learning_rate'],
                             n_estimators=int(xgboostBO.max['params']['n_estimators']),
                             min_child_weight=xgboostBO.max['params']['min_child_weight'],
                             subsample=xgboostBO.max['params']['subsample'],
                             colsample_bytree=xgboostBO.max['params']['colsample_bytree'],
                             random_state = 0)
                             
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)    ## 혼동행렬
xgb_model.score(X_train, y_train)   ## train data accuracy
xgb_model.score(X_test, y_test)     ## test data accuracy
```

:small_blue_diamond: **파이썬 코드 - 변수 중요도 시각화**

```python
from xgboost import plot_importance
import matplotlib.pyplot as plt
%matplotlib inline
fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(xgb_model, ax=ax)
```
