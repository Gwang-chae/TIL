# numpy

* numpy : Numerical Python를 지칭
* numpy는 vactor와 matrix 연산에 특화
* Pandas와 Matplotlib의 기반이 되는 module
* Machine Learning, Deep Learning에서 많이 사용됨



## numpy 특징 (python의 list와 비교)

* python의 list와 상당히 유사
* 차이가 있다면 python의 list는 다른 데이터 타입을 같이 list 안에 저장이 가능하지만
* numpy의 ndarry는 모두 같은 데이터 타입을 사용해야 함
* python의 list보다 메모리 효율이나 실행속도면에서 유리

``` python
# python의 list
a = [1,2,3,4,5] # python의 list
print(a)        # [1, 2, 3, 4, 5] => list literal
print(type(a))  # <class 'list'>

# numpy의 ndarray
arr = np.array([1,2,3,4,5])
print(arr)       # [1 2 3 4 5] => ndarray의 literal
print(type(arr)) # <class 'numpy.ndarray'>
print(arr.dtype) # int32(int:정수, 32:32bit)
print(arr[0])    # 1
print(type(arr[0])) # <class 'numpy.int32'>

list = [100,3.14,True,'Hello']
print(list)		# [100, 3.14, True, 'Hello']

arr = np.array([100,3.14,True,'Hello'])
print(arr)		# ['100' '3.14' 'True' 'Hello']
```



## numpy 특징 (다차원 속성)

* `ndim()` 을 통해 numpy  차원 확인 가능

```python
a = [1,2,3,4]
arr = np.array(a, dtype=np.float64)
print(arr)
print(arr.ndim)  # ndim => 차원의 개수 : 1
```

* `shape()`를 통해 차원의 수와 차원의 요소를 튜플로 표현

``` python
print(arr.shape) # shape => 차원의 개수와 각 차원의 요소를 tuple로 표현
                 # (4,)
    
# 3차원 중첩 리스트
a = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
arr = np.array(a, dtype=np.float64)
print(arr.shape) # (2, 3, 3)
```



## ndarray의 크기 확인

* `size()`를 통해 ndarray의 크기 확인가능

``` python
a = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
arr = np.array(a, dtype=np.float64)
print(arr.size)  # 12
print(len(arr))  # 첫번째 차원의 요소 개수를 리턴 -> 4
```



## ndarray의 데이터 타입 변경

* `astype()`으로 ndarray의 데이터 타입을 변경해 줄 수 있음

```python
arr = np.array([1.5, 2.3, 8.3, 9.8, 7.7], dtype=np.float64)
print(arr)				# [1.5 2.3 8.3 9.8 7.7]

result = arr.astype(np.int32)
print(result)			# [1 2 8 9 7]
print(result.dtype)		# int32
```



## ndarray 생성 방법

* `zeros()` : 0으로 채운 ndarray 생성

``` python
arr = np.zeros((3,4))   # 0으로 채운 numpy array 생성
                        # 인자로 shape를 명시
                        # dtype은 np.float64로 지정
```

* `ones()` : 1로 채운 ndarray 생성

```python
arr = np.ones((2,5))    # 1로 채운 numpy array 생성
						# 인자로 shape를 명시
                        # dtype은 np.float64로 지정
```

* `full`() : 지정한 값으로 ndarray 생성

```python
arr = np.full((3,5), 7, dtype=np.float64) # 7로 채운 numpy array 생성
										  # 인자로 shape를 명시
```

* `_like()` : 인자로 받는 array shape과 같은 shape의 array 생성

```python
result = np.zeros_like(arr, dtype=np.float64)    # 위에서 만든 array를 인자로 주면
                                                 # 그 인자의 shape과 같은 shape의 													 ndarray를 0으로 채워 생성

result = np.ones_like(arr, dtype=np.float64)     # 위에서 만든 array를 인자로 주면
                                                 # 그 인자의 shape과 같은 shape의 													 ndarray를 1으로 채워 생성
```

* `arange()` : 주어진 범위 내에서 지정한 간격으로 연속적인 원소를 가진 ndarray 생성

```python
# pythond의 range와 상당히 유사
# 주어진 범위 내에서 지정한 간격으로 연속적인 원소를 가진 ndarray 생성

arr = np.arange(1,10,1)
print(arr)		# [1 2 3 4 5 6 7 8 9]
				# list와는 다르게 ,로 구분되지 않음
```

* `linspace()` : start부터 stop까지의 범위에서 

  ​				     num개의 숫자를 균일한 간격으로 생성하여 ndarray 생성

```python
# linspace 기능을 확인하기 위해 그래프로 데이터를 표현
import matplotlib.pyplot as plt

# np.linspace(start, stop, num)
# start부터 stop까지의 범위에서 num개의 숫자를 균일한 간격으로
# 데이터를 생성해서 ndarray를 만드는 함수

arr = np.linspace(1,121,31)
# 원소간의 간격은 (stop-start) / (num-1)
# (121-1) / (31-1)
print(arr)			# [1. 5. 9. ...]

plt.plot(arr, '*')
plt.show()
```



## ndarray를 랜덤값 기반으로 생성

* `random.normal()` : 정규분포 확률밀도함수에서 실수 표본을 추출해서 ndarray 생성

``` python
mean = 50
std = 2
arr = np.random.normal(mean, std, (100000,))
print(arr)
plt.hist(arr, bins=100)
plt.show()
```

* `random.rand(d0,d1,d2, ...)` : [0,1)의 범위에서 실수를 추출, 균등분포로 추출

```python
arr = np.random.rand(100000)
print(arr)
plt.hist(arr,bins=100)
plt.show()
```

* `random.randn(do,d1,d2, ...)` : 실수 추출, 표준정규분포에서 난수 추출,

  ​															 평균이 0, 표준편차가 1

```python
arr = np.random.randn(100000)
print(arr)
plt.hist(arr, bins=100)
plt.show()
```

* `random.randint()` : 정수 추출, 균등분포 확률밀도함수에서 난수를 추출

```python
arr = np.random.randint(-100,100,(100000,))
print(arr)
plt.hist(arr, bins=100)
plt.show()
```

* `random.random(shape)` : [0,1)의 범위에서 실수를 추출, 균등분포로 추출

```python
arr = np.random.random((100000,))
print(arr)
plt.hist(arr, bins=100)
plt.show()
```



## numpy가 제공하는 랜덤 관련 함수

* 난수 재현
  * 랜덤값도 실제로는 특정 알고리즘의 결과물
  * 초기 시작값을 설정해주면 항상 같은 랜덤값이 도출

``` python
np.random.seed(10)						# seed를 설정해주면 같은 값이 계속 출력
arr = np.random.randint(0,100,(10,))
print(arr)
```

* ndarray의 순서를 랜덤하게 변형 - `shuffle()` 이용

``` python
arr = np.arange(10)
np.random.shuffle(arr) # ndarray 자체가 변형
print(arr)
```

* ndarray안에서 일부를 무작위로 선택하는 기능 - `choice`() 이용

``` python
np.random.choice(arr, size, replace, p)
# arr : numpy array 혹은 정수가 가능, 정수일 때는 arange(정수)로 들어가는 것
# size : 정수값, 샘플의 숫자
# replace : Boolean
#           True일 때, 데이터를 중복 샘플링 가능
# p : ndarray, 각 데이터가 샘플링 될 확률을 가지고 있는 ndarray

arr = np.random.choice(5, 10, replace=True, p=[0.2, 0, 0.4, 0.1, 0.3])
print(arr)
```



## ndarray의 shape 속성 조정

* `shape()` 을 이용해서 ndarray를 변경하는 것은 바람직하지 않음



#### 1. `reshape()`

* 따라서, 보통 `reshape()` 함수를 사용해서 처리
* 이 때, `reshape()`는 새로운 ndarray를 만드는게 아니라 view를 생성하는 것
* view가 아닌 진짜 shape가 바뀐 ndarray를 만드려면 `copy`를 사용

``` python
arr = np.arange(0,12,1)
arr1 = arr.reshape(4,3)  # 새로운 ndarray를 만드는 게 아니라 View를 생성

print(arr)
print(arr1)

arr[0] = 100
print(arr)			# [100 1 2 3 4 5 6 7 8 9 10 11]
print(arr1)			# [[100 1 2]
 					#  [ 3 4 5]
 					#  [ 6 7 8]
 					#  [ 9 10 11]]

arr = np.arange(0,12,1)
arr1 = arr.reshape(4,3).copy() # View가 아니라 진짜 shape을 바꾼 ndarray를 만드려면 								   copy() 함수 사용
```



#### 2. `ravel()`

* 추가적으로 `ravel()` 함수를 통해 ndarray 모든 요소를 1차원 vector로 리턴 가능

``` python
arr = np.arange(0,10,1).reshape(5,-1).copy()
print(arr)			# [[0 1 2 3 4]
 					#  [5 6 7 8 9]]
    
arr1 = arr.ravel()
print(arr1)			# [0 1 2 3 4 5 6 7 8 9]
```



#### 3. `resize()`

* shape를 바꾸는데 `resize()`도 사용할 수 있음
* `resize()`는 View를 생성하는 것이 아닌 원본의 shape을 바꿔줌
* `reshape()`과의 차이점
  * `resize()`는 요소 개수가 맞지 않아도 shape이 변경
* `resize()`는 요소 수가 줄어들면 기존 데이터를 버리고 남으면 0으로 세팅

```python
import numpy as np
np.random.seed(10)

arr = np.random.randint(0,10,(3,4))		# 3x4 행렬 생성
print(arr)

# arr.resize(2,6)   # View가 생성되지 않음 원본이 shape으로 바뀜
# print(arr)        

arr.resize(3,5)   # reshape()은 요소 개수가 맞지 않으면 reshape이 안됨
                  # resize()는 요소 개수가 맞지 않아도 shape이 변경
print(arr)

arr.resize(2,2)   # 요소 수가 줄어들면 기존 데이터를 버리고 남으면 0으로 세팅
print(arr)
```

​	

## Indexing & Slicing

* `ndarray`는 `python`의 `list`와 마찬가지로 `indexing`과 `slicing`이 가능
* 보통 `for`문을 통해 각 요소를 출력하는데 이때 `enumerate()`를 사용하면 `index`까지 편리하게 뽑아낼 수 있음

```python
import numpy as np
arr = np.arange(10,20,1)
print(arr)					# [10 11 12 13 14 15 16 17 18 19]

# ndarray의 각 요소 출력(일반적인 for문 사용)
for i in arr :
    print(i)

# enumerate() 사용
for (idx,j) in enumerate(arr):
    print('인덱스 : {}, 데이터 : {}'.format(idx,j))

# python list와 유사하게 indexing, slicing 가능
print(arr[3]) 				 # 13
print(arr[1:4]) 			 # [11 12 13]
print(arr[:-1])				 # [10 11 12 13 14 15 16 17 18]
print(arr[1:-1:2])			 # [11 13 15 17]
```



## Boolean indexing & Fancy indexing

#### 1. Boolean indexing

* `boolean indexing`은 `ndarray`의 각 요소의 선택여부

* `True`, `False`로 구성된 `boolean mask`를 이용하여 지정하는 방식
* `boolean mask`의 `True`에 해당하는 `index`만을 조회하는 방식

``` python
import numpy as np

arr = np.random.randint(0,10,(5,))
print(arr)					# [7 7 0 2 2]
print(arr%2)				# [1 1 0 0 0]
print(arr%2 == 0)			# [False False  True  True  True]
print(arr[arr%2 == 0])		# [0 2 2]
```



#### 2. Fancy indexing

* `ndarray`에 `index`배열을 전달하여 배열요소를 참조하는 방식

``` python
import numpy as np

arr = np.arange(0,12,1).reshape(3,4).copy()
print(arr)					# [[ 0  1  2  3]
							#  [ 4  5  6  7]
							#  [ 8  9 10 11]]
print(arr[2,2])				# 10
print(arr[1:2,2])			# [6]
print(arr[1:2,1:2])			# [[5]]
print(arr[[0,2],2])			# [ 2 10]
print(arr[[0,2],2:3])		# [[ 2]
 							#  [10]]
    
# 예제
# [[1 3]
#  [9 11]] 를 추출하시오

# 해결방법1
print(arr[[0,2]][:,[1,3]])

# 해결방법2 - numpy 제공 함수 사용
print(arr[np.ix_([0,2],[1,3])])
	## 단순히 arr[[0,2],[1,3]]을 실행하면 [1 11] 값만 얻게됨
    ## 따라서 numpy가 제공하는 ix_() 함수를 실행
```



## ndarray의 연산, 행렬곱, 전치행렬 

#### 1. `+`,`-` 연산

* `python`의 `list`에서 `+` 연산자는 `concatenation`이다.
* 그러나 `ndarray`에서 `+` 연산자는 `vector` 또는 `matrix` 연산
* 따라서, `ndarray`의 사칙연산 기본 전제는 연산하려는 `ndarray`의 `shape`이 같다는 것

``` python
import numpy as np

arr1 = np.array([[1,2,3],[4,5,6]])
arr2 = np.array([[7,8,9],[10,11,12]])
print(arr1 + arr2)		# [[ 8 10 12]
						#  [14 16 18]]
```

* `shape`이 안 맞는 경우, `ndarray`는 `broadcasting` 수행

``` python
import numpy as np

arr1 = np.array([[1,2,3],[4,5,6]])
# arr2를 schema로 줄 경우
arr2 = 3
print(arr1 + arr2)		# [[4 5 6]
						#  [7 8 9]]
```



#### 2. 비교연산

* `+` ,`-` 연산과 마찬가지로 비교연산도 같은 `index`끼리 수행

* `boolean mask`로 표현

``` python
import numpy as np

arr1 = np.arange(10)
arr2 = np.arange(10)
print(arr1)							# [0 1 2 3 4 5 6 7 8 9]
print(arr2)							# [0 1 2 3 4 5 6 7 8 9]

# '_equal()'함수로 ndarray간의 동일 여부 파악 가능
print(np.array_equal(arr1,arr2))	# True
```





#### 3. 행렬곱

* 두 행렬간의 행렬곱은 `np.dot(`),` np.matmul()`로 수행 가능

* `np.dot(A,B)`에서 A행렬의 열 `vector`와 B행렬의 행 `vector`의 size가 같아야 함

* 만약에 크기가 다르면 `reshape(`)나 `resize()`로 크기를 맞추고 연산을 수행해야 함

``` python
import numpy as np

arr1 = np.array([[1,2,3],[4,5,6]])
arr2 = np.array([[7,8],[9,10],[11,12]])
print(arr1)					# [[1 2 3]
							#  [4 5 6]]
print(arr2)					# [[ 7  8]
							#  [ 9 10]
    						#  [11 12]]
print(np.dot(arr1,arr2))	# [[ 58  64]
							#  [139 154]]
```

* 행렬곱 연산이 존재해야 다양한 크기의 행렬을 연속적으로 이용해서 작업을 수행 가능



#### 4. 전치행렬(Transpose)

* 전치행렬이란 기존 행렬의 행과 열의 위치를 바꾼 행렬
* 전치행렬의 표현은 윗첨자로 T를 쓴다.
* 전치행렬로 만들기 위해서는 `ndarray` 뒤에 `.T`를 붙여준다.

``` python
arr = np.array([[1,2,3],[4,5,6]])
print(arr)				# [[1,2,3]
						#  [4,5,6]]
t_array = arr.T	
print(t_array)			# [[1 4]
						#  [2 5]
    					#  [3 6]]
arr[0,0] = 100
# arr.T는 View라는 것을 알 수 있음
print(t_array)  		# [[100   4]
						#  [  2   5]
    					#  [  3   6]]

```



## numpy iterator

*  `ndarray`의 각 요소를 출력하려 할 때에 차원이 높아지게 되면 

  기존의 `for`문을 차원에 맞게 중첩해야한다.

* 이런 수고를 덜기 위해 numpy의 `iterator`를 사용

``` python
import numpy as np

# 1차원 ndarray의 각 요소 출력
arr = np.array([1,2,3,4,5])
it = np.nditer(arr, flags=['c_index']) # 1차원일 때 flags값으로 'c_index'를 줌

while not it.finished:    # iterator가 지정하는 위치가 끝이 아닐동안 반복
    idx = it.index         # iterator가 가리키는 곳의 index 숫자를 가져옴
    print(arr[idx], end=' ')
    it.iternext()          # iterator를 다음 요소로 이동
    
# 다차원 ndarray의 각 요소 출력
arr = np.array([[1,2,3],[4,5,6]])
it = np.nditer(arr, flags=['multi_index'])  # 다차원인 경우에는 multi_index를 flag로 줌

while not it.finished:    # iterator가 지정하는 위치가 끝이 아닐동안 반복
    idx = it.multi_index         # iterator가 가리키는 곳의 index 숫자를 가져옴
    print(idx)				# it.multi_index 값을 확인해보면 index가 어떻게 들어오는지 확인 가능
    						# (0, 0)
							# 1 (0, 1)
							# 2 (0, 2)
							# 3 (1, 0)
							# 4 (1, 1)
							# 5 (1, 2)
    print(arr[idx], end=' ')
    it.iternext()
```



## 집계함수와 축(axis)

#### 1. 집계함수

* `numpy`는 다양한 집계함수를 제공

```python
import numpy as np

arr = np.array([1,2,3],[4,5,6])
print(arr)				# [[1 2 3]
						#  [4 5 6]]
    
print(np.sum(arr)) 		# 요소의 총합, print(arr.sum())와 동일
						# 21
    
print(np.cumsum(arr)) 	# 누적합을 1차원 vector형태로 출력
						# [ 1  3  6 10 15 21]
    
print(np.mean(arr))		# 평균 3.5
	
print(np.max(arr)) 		# 최대값 6

print(np.min(arr)) 		# 최소값 1

print(np.argmax(arr)) 	# 최대값을 찾아서 최대값의 순번(index)을 리턴 5

print(np.argmin(arr)) 	# 최소값을 찾아서 최소값의 순번(index)을 리턴 0

print(np.std(arr)) 		# 표준편차 1.707	

print(np.exp(arr)) 		# 자연상수
print(np.log10(arr)) 	# 로그값

```



#### 2. 축(axis)

* `numpy`의 모든 집계함수는 `axis`를 기준으로 계산됨
* 만약 `axis`를 지정하지 않으면, `axis`는`None`으로 설정

​        => 함수의 대상범위를 전체 `ndarray`로 지정하게 됨

``` python
import numpy as np


# 3차원의 경우
np.random.seed(1)

arr = np.random.randint(0,10,(2,2,3))
print(arr)				# [[[5 8 9]
						#   [5 0 0]]
    
						#   [[1 7 6]
						# 	[9 2 4]]]
print(arr.sum())		# 56
            
print(arr1.sum(axis=0))  # 3차원에서 axis=0 => depth방향
						 # [[ 6 15 15]
						 #  [14  2  4]]
        
print(arr1.sum(axis=1))  # 3차원에서 axis=1 => 행방향! 세로방향!!
						 # [[10  8  9]
     					 # [10  9 10]]
        
print(arr1.sum(axis=2))  # 3차원에서 axis=2 => 열방향! 가로방향!!
						 # [[22  5]
						 # [14 15]]

## 1차원의 경우 axis=0 => 열방향! 가로방향!!
## 2차원의 경우 axis=0 => 행방향! 세로방향!!
##			  axis=1 => 열방향! 가로방향!!
```



## numpy 정렬

* `numpy`는 `axis`를 기준으로 정렬하는 `sort()` 함수 제공
* 만약 `axis`를 지정하지 않으면 -1 값으로 지정
* `np.sort()` : 정렬된 결과 `ndarray`를 출력
* `arr.sort()` : 원본을 정렬, return 값은 None

``` python
import numpy as np

arr = np.arange(10)
np.random.shuffle(arr)
print(arr)					# [9 6 1 0 8 7 3 4 2 5]

print(np.sort(arr)) # 오름차순으로 정렬 (default가 오름차순)
					# [0 1 2 3 4 5 6 7 8 9]

# ndarray는 특수한 indexing 제공 => 역순으로 정렬하기 위한 indexing 제공
print(np.sort(arr)[::-1])	# [9 8 7 6 5 4 3 2 1 0]
```



## numpy 요소 추가, 삭제

#### 1. `concatenate()`

* `python` `list` 사용할때 원소를 추가하려면 => `append()`
* `ndarray`를 쓸 때 원소를 추가할 때 보통 => `concatenate()`

``` python
import numpy as np

arr = np.array([[1,2,3],[4,5,6]])
new_row = np.array([7,8,9])

# arr에 new_row vector를 행으로 붙임

result = np.concatenate((arr,new_row.reshape(1,3)),axis=0)
print(result)			# [[1 2 3]
 						# [4 5 6]
						# [7 8 9]]
```



#### 2. `delete()`

* `axis`를 기준으로 행과 열을 삭제 가능
* 만약 `axis`를 지정하지 않으면 1차원배열로 변환 후 삭제

* 원본은 변경하지 않고 처리가 된 새로운 배열을 return

``` python
import numpy as np

np.random.seed(1)

arr = np.random.randint(0,10,(3,4))
print(arr)					# [[5 8 9 5]
							# [0 0 1 7]
							# [6 9 2 4]]

result = np.delete(arr,1) 	# axis가 설정되지 않았기 때문에 1차원배열로 변환 후 1번 인덱스 삭제
print(result)             	# [5 9 5 0 0 1 7 6 9 2 4]

result = np.delete(arr, 1, axis=0) 
print(result)               # [[5 8 9 5]
                            #  [6 9 2 4]]
```

