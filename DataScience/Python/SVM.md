# Support Vector Machine

- 데이터 클래스의 결정 경계를 정의하는 지도학습 모델
- 경계를 기준으로 어느 쪽에 속하는지 확인하여 분류 과제 수행 가능
- 허용 가능한 오차 범위 내에서 가능한 최대 마진을 만드는 모델
- 큰 마진의 결정 경계는 일반화 오차를 낮추는 경향이 있음</br></br>
:bulb: **마진**은 클래스를 구분하는 결정 경계와 이 결정 경계에 가장 가까운 훈련 샘플 사이의 거리로 정의함</br></br>

:small_orange_diamond: **Linear Kernel로 선형 문제 다루기**
- 데이터 분류를 위한 결정 경계로 선형 결정 함수를 찾는 것
- feature 개수가 2개라면 결정 함수가 선으로 나타나고 3개라면 결정 함수가 평면으로 나타남
- 선형계획법 강의에서 공부한 최적화 방법으로 결정 함수를 찾음 (라그랑제 승수법, 쌍대문제 등)</br></br>
<img src="https://user-images.githubusercontent.com/68538876/102189662-52fd2500-3efa-11eb-9334-58f53b4bb338.png" width="525" height="225"></br></br>  


:small_orange_diamond: **슬랙 변수를 사용하여 비선형 문제 다루기**
- 소프트 마진 분류
- 슬랙 변수는 선형적으로 구분되지 않는 데이터에서 선형 제약 조건을 완화할 필요가 있기 때문에 도입됨
- 적절히 비용을 손해보면서 분류 오차가 있는 상황에서 최적화 알고리즘이 수렴함
- 변수 C를 통해 분류 오차를 어느정도 허용할 것인지 지정할 수 있음
- C 값이 크면 오차를 허용하지 않고 값이 작으면 분류 오차에 덜 엄격해짐
- C 값을 사용하여 마진 폭을 제어할 수 있음</br></br>
<img src="https://user-images.githubusercontent.com/68538876/102190021-cef76d00-3efa-11eb-97ce-be982a3aa18d.png" width="525" height="225"></br></br>  



:small_orange_diamond: **커널 SVM을 사용하여 비선형 문제 다루기**
- 커널 방법은 매핑 함수를 사용하여 원본 특성의 비선형 조합을 선형적으로 구분되는 고차원 공간에 투영하는것
- 고차원 공간에서 두 클래스를 구분하는 선형 초평면은 원본 특성 공간으로 되돌리면 비선형 결정 경계가 됨
- 매핑함수를 사용하여 훈련 데이터를 고차원 특성 공간으로 변환한 후, 새로운 특성 공간에서 데이터를 분류하는 선형 SVM 모델을 훈련함.
- 동일한 매핑함수를 사용하여 새로운 데이터를 변환하고 선형 SVM 모델을 사용하여 분류할 수 있음
- 커널 함수는 두 점 사이 점곱을 계산하는 데 드는 높은 비용을 절감하기 위해 정의됨
- 가장 널리 사용되는 커널 중 하나는 방사기저함수 = 가우시안 커널</br></br>
<img src="https://user-images.githubusercontent.com/68538876/102190495-72488200-3efb-11eb-83a6-d82bca66947a.png" width="455" height="293"></br>  
<img src="https://user-images.githubusercontent.com/68538876/102190338-4200e380-3efb-11eb-8fe6-64a061bddf23.png" width="320" height="225"><img src="https://user-images.githubusercontent.com/68538876/102190812-d703dc80-3efb-11eb-9f4e-1b71a73f6937.png" width="320" height="225"></br></br>


:small_blue_diamond: **파이썬 코드 - Linear Kernel 로 선형문제 다루기**
```python
### linear SVM ###
from sklearn.svm import LinearSVC
from sklearn import metrics
model = LinearSVC()
model.fit(x_train, y_train)
coef = model.coef_[0]
intercept = model.intercept_[0]
print('계수 : ', coef)
print('상수 : ', intercept)

print("----- train score -----")
y_pred = model.predict(x_train)
print('accuracy : ', model.score(x_train, y_train))
print('recall : ', metrics.recall_score(y_train, y_pred))
print('f1 score : ', metrics.f1_score(y_train, y_pred))
print('f2 score : ', metrics.fbeta_score(y_train, y_pred, beta=2))

print("----- test score -----")
y_pred = model.predict(x_test)
print('accuracy : ', model.score(x_test, y_test))
print('recall : ', metrics.recall_score(y_test, y_pred))
print('f1 score : ', metrics.f1_score(y_test, y_pred))
print('f2 score : ', metrics.fbeta_score(y_test, y_pred, beta=2))


### visualization ###

x_train = pd.DataFrame(x_train, columns=['pc1','pc2', 'pc3'])
x_test = pd.DataFrame(x_test, columns=['pc1','pc2', 'pc3'])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d

### train
figure = plt.figure()
ax = Axes3D(figure, elev=-160, azim=-10)
mask = y_train == 0
ax.scatter(x_train.iloc[mask, 0], x_train.iloc[mask, 1], x_train.iloc[mask, 2], c='b')
ax.scatter(x_train.iloc[~mask, 0], x_train.iloc[~mask, 1], x_train.iloc[~mask, 2], c='r')
ax.set_xlabel("pc1")
ax.set_ylabel("pc2")
ax.set_zlabel("pc3")

figure = plt.figure()
ax = Axes3D(figure, elev=-160, azim=-10)
xx = np.linspace(x_train.iloc[:, 0].min() - 0.8, x_train.iloc[:, 0].max() + 0.8, 50)
yy = np.linspace(x_train.iloc[:, 1].min() - 0.8, x_train.iloc[:, 1].max() + 0.8, 50)

XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
mask = y_train == 0
ax.scatter(x_train.iloc[mask, 0], x_train.iloc[mask, 1], x_train.iloc[mask, 2], c='b')
ax.scatter(x_train.iloc[~mask, 0], x_train.iloc[~mask, 1], x_train.iloc[~mask, 2], c='r')
ax.set_xlabel("pc1")
ax.set_ylabel("pc2")
ax.set_zlabel("pc3")
plt.show()

### test
figure = plt.figure()
ax = Axes3D(figure, elev=-160, azim=-10)
mask = y_test == 0
ax.scatter(x_test.iloc[mask, 0], x_test.iloc[mask, 1], x_test.iloc[mask, 2], c='b')
ax.scatter(x_test.iloc[~mask, 0], x_test.iloc[~mask, 1], x_test.iloc[~mask, 2], c='r')
ax.set_xlabel("pc1")
ax.set_ylabel("pc2")
ax.set_zlabel("pc3")

figure = plt.figure()
ax = Axes3D(figure, elev=-160, azim=-10)
xx = np.linspace(x_test.iloc[:, 0].min() - 0.8, x_test.iloc[:, 0].max() + 0.8, 50)
yy = np.linspace(x_test.iloc[:, 1].min() - 0.8, x_test.iloc[:, 1].max() + 0.8, 50)

XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
mask = y_test == 0
ax.scatter(x_test.iloc[mask, 0], x_test.iloc[mask, 1], x_test.iloc[mask, 2], c='b')
ax.scatter(x_test.iloc[~mask, 0], x_test.iloc[~mask, 1], x_test.iloc[~mask, 2], c='r')
ax.set_xlabel("pc1")
ax.set_ylabel("pc2")
ax.set_zlabel("pc3")
plt.show()

### Plane : ax+by+cz+d=0 ###
a = coef[0]
b = coef[1]
c = coef[2]
d = intercept
```

:small_blue_diamond: **파이썬 코드 - 슬랙 변수를 사용하여 비선형 문제 다루기**
```python
from sklearn.svm import SVC
model = SVC(kernel='linear', C=1.0, random_state=0)
model.fit(x_train, y_train)
coef = model.coef_[0]
intercept = model.intercept_[0]
print('계수 : ', coef)
print('상수 : ', intercept)
```

:small_blue_diamond: **파이썬 코드 - 커널 SVM을 사용하여 비선형 문제 다루기**
```python
from sklearn.svm import SVC
model = SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0)
model.fit(x_train, y_train)

## kernel rbf: 방사기저함수 = 가우시안 커널
## gamma: 가우시안 구의 크기를 제한하는 매개변수. 이 값을 크게 하면 학습 데이터에 많이 의존해서 결정 경계가 구불구불해지고 오버피팅을 초래할 수 있음.
```
