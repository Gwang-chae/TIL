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

* `ndim` 을 통해 numpy  차원 확인 가능

```python
a = [1,2,3,4]
arr = np.array(a, dtype=np.float64)
print(arr)
print(arr.ndim)  # ndim => 차원의 개수 : 1
```

* `shape`를 통해 차원의 수와 차원의 요소를 튜플로 표현

``` python
print(arr.shape) # shape => 차원의 개수와 각 차원의 요소를 tuple로 표현
                 # (4,)
    
# 3차원 중첩 리스트
a = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
arr = np.array(a, dtype=np.float64)
print(arr.shape) # (2, 3, 3)
```



## ndarray의 크기 확인

* `size`를 통해 ndarray의 크기 확인가능

``` python
a = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
arr = np.array(a, dtype=np.float64)
print(arr.size)  # 12
print(len(arr))  # 첫번째 차원의 요소 개수를 리턴 -> 4
```



## ndarray의 데이터 타입 변경

* `astype`으로 ndarray의 데이터 타입을 변경해 줄 수 있음

```python
arr = np.array([1.5, 2.3, 8.3, 9.8, 7.7], dtype=np.float64)
print(arr)				# [1.5 2.3 8.3 9.8 7.7]

result = arr.astype(np.int32)
print(result)			# [1 2 8 9 7]
print(result.dtype)		# int32
```



# ndarray 생성 방법

* `zeros` : 0으로 채운 ndarray 생성

``` python
arr = np.zeros((3,4))   # 0으로 채운 numpy array 생성
                        # 인자로 shape를 명시
                        # dtype은 np.float64로 지정
```

* `ones` : 1로 채운 ndarray 생성

```python
arr = np.ones((2,5))    # 1로 채운 numpy array 생성
						# 인자로 shape를 명시
                        # dtype은 np.float64로 지정
```

* `full` : 지정한 값으로 ndarray 생성

```python
arr = np.full((3,5), 7, dtype=np.float64) # 7로 채운 numpy array 생성
										  # 인자로 shape를 명시
```

* `_like` : 인자로 받는 array shape과 같은 shape의 array 생성

```python
result = np.zeros_like(arr, dtype=np.float64)    # 위에서 만든 array를 인자로 주면
                                                 # 그 인자의 shape과 같은 shape의 													 ndarray를 0으로 채워 생성

result = np.ones_like(arr, dtype=np.float64)     # 위에서 만든 array를 인자로 주면
                                                 # 그 인자의 shape과 같은 shape의 													 ndarray를 1으로 채워 생성
```

* `arange` : 주어진 범위 내에서 지정한 간격으로 연속적인 원소를 가진 ndarray 생성

```python
# pythond의 range와 상당히 유사
# 주어진 범위 내에서 지정한 간격으로 연속적인 원소를 가진 ndarray 생성

arr = np.arange(1,10,1)
print(arr)		# [1 2 3 4 5 6 7 8 9]
				# list와는 다르게 ,로 구분되지 않음
```

* `linspace` : start부터 stop까지의 범위에서 

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

* ndarray의 순서를 랜덤하게 변형 - `shuffle` 이용

``` python
arr = np.arange(10)
np.random.shuffle(arr) # ndarray 자체가 변형
print(arr)
```

* ndarray안에서 일부를 무작위로 선택하는 기능 - `choice` 이용

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



## ndarray의 shape 속성

* shape 속성의 값을 바꾸어서 ndarray를 변경하는 것은 바람직하지 않음
* 따라서, `reshape()` 함수를 사용해서 처리
* 이 때, `reshape`는 새로운 ndarray를 만드는게 아니라 view를 생성하는 것
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

* 추가적으로 `ravel()` 함수를 통해 ndarray 모든 요소를 1차원 vector로 리턴 가능

``` python
arr = np.arange(0,10,1).reshape(5,-1).copy()
print(arr)			# [[0 1 2 3 4]
 					#  [5 6 7 8 9]]
    
arr1 = arr.ravel()
print(arr1)			# [0 1 2 3 4 5 6 7 8 9]
```

