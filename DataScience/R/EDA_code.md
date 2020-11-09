# 탐색적자료분석 R 코드 (Exploratory Data Analysis)
###### 동국대학교 통계학과 `탐색적 자료 분석' 전공 강의에서 공부하며 정리한 내용 정리
###### ***기본적인 R코드 문법 정리 => [Rcode_basic](https://github.com/datajudy20/TIL/blob/main/DataScience/R/Rcode_basic.pdf)***
---
## 기초 코드
- 모든 함수 정보 확인 : **help.start**
- WorkDirectory 설정 : **setwd()**
- 특정 조건 만족하는 데이터 추출 : **subset(data, 조건)**
- 그룹별로 통계량 출력 : **by(y, group, summary)**
- 파일 불러올 때 NA 표시 옵션 : **na.string=‘ ’**<BR/><BR/>
## 기본 코드
#### ```grep(pattern, vec, value)``` : 특정 패턴 만족하는 값 또는 위치 반환
- pattern 옵션
  + ^a : a로 시작  
  + a$ : a로 끝  
  + a : a 포함  
- value 옵션
  + True : 값 반환
  + False : 위치 반환
#### ```substr(vec, start, stop)``` : 일부 문자 추출
- vec : 문자형 벡터
- start : 추출할 문자의 첫번째 위치
- stop : 추출할 문자의 마지막 위치
#### ```sort(vec, decreasing)``` : 벡터 정렬
- 순서대로 값을 반환
- 데이터 프레임 정렬할 때 사용 불가
#### ```order(vec, decreasing)``` : 벡터 정렬
- 순서대로 인덱스를 반환
- 데이터 프레임 정렬할 때 사용 가능
- ```order(data$변수1, -data$변수2)``` : 데이터프레임의 변수 1은 오름차순, 변수2는 내림차순으로 정렬<BR/><BR/>
### 질적자료
#### ```table(vec)``` : 질적 자료 빈도표
- useNA='ifany' 옵션 : NA 값 빈도수도 출력함
#### ```sort(table(), decreasing)``` : 빈도 결과 정렬
#### ```prop.table(table(x))``` = ```table(x)/sum(table(x))```: 질적 자료 백분율<BR/><BR/>
### 그래프
#### ```par(mfrow=c(,))``` : 하나의 창에 여러 그래프 넣을 때 (Python의 subplot과 비슷)
#### ```abline(v= , h= , lty= )``` : 그래프 위에 직선 그릴 때
#### ```legend(위치, vec, col= )``` : 그래프에 범례 넣을 때
#### ```barplot(table(), ...)``` : 질적자료 - 막대 그래프
#### ```pie(table(), ...)``` : 질적자료 - 원 그래프
#### ```hist(vec, breaks)``` : 양적자료 - 히스토그램
#### ```boxplot(y~group, data)``` : 양적자료 - 상자 그림
#### ```plot(x, y)``` : 양적자료 - 산점도<BR/><BR/>
### 함수 적용
#### ```apply(data, margin, function)``` : data에 함수 적용
- margin = 1 : 행끼리 함수 적용
- margin = 2 : 열끼리 함수 적용
#### ```tapply(vec, factor, function)``` : factor의 수준별로 함수 적용
#### ```cut(vec, breaks= , labels= )``` : 수치형 벡터를 순서형 factor로 변환<BR/><BR/>
## 군집화 코드 - Clustering
#### Hierarchical Clustering
```R
dist(df)              # 유클리드 거리. method 옵션으로 변경 가능함
hclust(dist(df))      # clustering - complete 방식
plot(hclust())        # dendrogram
cutree(hclust(), k)   # k개의 그룹으로 자르기
```
#### Kmeans Clustering
```R
kmeans(df, centers)                 # centers 옵션으로 군집 개수 가정 후 clustering
silhouette(kmeans()$cl, dist())     # 실루엣으로 군집 개수 결정 - 값이 1에 가까울수록 좋음
plot(silhouette())
```  
## 주성분 분석 코드 - PCA
<img src="https://user-images.githubusercontent.com/68538876/98513589-71448500-22ab-11eb-83ea-346b501f2b94.png" width="450" height="110"><BR/>
```R
prcomp(df, scale)     # scale 옵션으로 correlation 이용
prcomp()$x            # 주성분으로 변환한 관측치의 PCA 값
prcomp()$rotation     # 주성분 계수
summary(prcomp())     # PCA의 설명 비율
screeplot(prcomp())
biplot(prcomp())
## sum(apply(df, 2, var)) = sum(eigen(cov(df))$values) = sum(df.PCA$sdev^2) = sum(diag(cov(df))) = sum(diag(cov(df.PCA$x)))
```  
## 특이값 분해 코드 - SVD
<img src="https://user-images.githubusercontent.com/68538876/98514606-14e26500-22ad-11eb-8cf3-18e988af8ea5.png" width="550" height="200"><BR/>
```R
## 두줄씩 같은 결과를 출력함
eigenvalue(cov(scale(X)))*(n-1)
svd(scale(X))$d^2

eigenvector(cov(scale(X))
svd(scale(X))$v
```  
## 요인 분석 코드 - FA
```R
factanal(df, k, rotation, scores)
str(factanal())
print(factanal(), sort)                       # mean(factanal()$uniquenesses) : factor가 설명하지 못하는 부분
plot(df, plot(loadings[,1], loadings[,2]))
```
