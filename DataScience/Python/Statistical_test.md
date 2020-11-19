# 두 그룹 사이 평균 차이 검정

:small_orange_diamond: **검정 절차**  
- 표본 크기가 30보다 작은 그룹에서 정규성 검정 진행 (shapiro-wilk test & Kolmogorov-Smirnov test)  
- 정규성을 만족하지 않는 변수에 대해서는 비모수 검정 진행 (mann-whitney test)  
- 정규성을 만족하는 변수에 대해서는 등분산성 검정 진행 (levene test)  
- 등분산성을 만족하는 변수에 대해서는 등분산 t test 진행
- 등분산성을 만족하지 않는 변수에 대해서는 이분산 t test 진행<BR/><BR/>  

:small_blue_diamond: **파이썬 코드 - 통계 검정 자동 클래스**
  ```python
  from scipy import stats
  
  x = data_A.copy()
  y = data_B.copy()
  var_list = list(data.columns)
  
  
  class TestAll:
    
    def __init__(self, var):
        self.var = var
        
    def test(self):
        shap_x = stats.shapiro(x[self.var])
        shap_y = stats.shapiro(y[self.var])
        
        ks_x = stats.kstest(x[self.var], 'norm')
        ks_y = stats.kstest(y[self.var], 'norm')
        
        if (((shap_x.pvalue >= 0.05)|(ks_x.pvalue >= 0.05)) & ((shap_y.pvalue >= 0.05)|(ks_y.pvalue >= 0.05))):
            print(str(self.var) + '변수는 정규성을 만족한다')
            self.levene()
        else:
            print(str(self.var) + '변수는 정규성을 만족하지 않는다')
            self.mann()
            
    def mann(self):
        test = stats.mannwhitneyu(x[self.var], y[self.var])
        
        if test.pvalue < 0.05:
            print('유의수준 0.05 하에서 ' + str(self.var) + '변수는 두 집단 사이에서 순위합이 같지 않다')
            self.boxplot_var(0.05)
        elif test.pvalue < 0.1:
            print('유의수준 0.1 하에서 ' + str(self.var) + '변수는 두 집단 사이에서 순위합이 같지 않다')
            self.boxplot_var(0.1)
        else:
            print(str(self.var) + '변수는 두 집단 사이에서 순위합이 같다', end='\n\n\n')
            print('-'*80)
            
    def levene(self):
        test = stats.levene(x[self.var], y[self.var])
        
        if test.pvalue < 0.05:
            print(str(self.var) + '변수는 두 집단 사이에서 분산이 같지 않다 => 이분산')
            self.Ttest(False)
        else:
            print(str(self.var) + '변수는 두 집단 사이에서 분산이 같다 => 등분산')
            self.Ttest(True)    

    def Ttest(self, tf):
        test = stats.ttest_ind(x[self.var], y[self.var], equal_var=tf) 
        
        if test.pvalue < 0.05:
            print('유의수준 0.05 하에서 ' + str(self.var) + '변수는 두 집단 사이에서 평균에 차이가 있다')
            self.boxplot_var(0.05)
        elif test.pvalue < 0.1:
            print('유의수준 0.1 하에서 ' + str(self.var) + '변수는 두 집단 사이에서 평균에 차이가 있다')
            self.boxplot_var(0.1)
        else:
            print(str(self.var) + '변수는 두 집단 사이에서 평균에 차이가 없다', end='\n\n\n')
            print('-'*80)

    def boxplot_var(self, alpha):
        plt.figure(figsize=(5, 7))
        plt.boxplot((x[self.var].to_numpy(), y[self.var].to_numpy()),
                    sym='o', labels=['0-40%', '70-100%'], meanline=True, showmeans=True)
        plt.title('{0} boxplot (alpha={1})'.format(self.var, alpha))
        plt.show()
        print('\n\n')
        print('-'*80)
        
        
  for i in range(len(var_list)):
    TestAll(var_list[i]).test()
  ```
