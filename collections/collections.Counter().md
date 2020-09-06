# collections.Counter()

* `Counter()` 는 인자로 iterable 또는 mapping 자료를 받음
* 주로 `list` , `dictionary` , `key = value` 형태, `문자열` 이 인자로 들어옴

```python
import collections
# list 
list = ['a','a','a','b','b','c']
print(collections.Counter(list))			# Counter({'a': 3, 'b': 2, 'c': 1})

# dictionary
dictionary = {'a': 3, 'b': 2, 'c': 1}
print(collections.Counter(dictionary))		# Counter({'a': 3, 'b': 2, 'c': 1})

# key = value 형태
print(collections.Counter(a=3, b=2, c=1))	# Counter({'a': 3, 'b': 2, 'c': 1})

# 문자열
string = 'abcaba'
print(collections.Counter(string))			# Counter({'a': 3, 'b': 2, 'c': 1})
```

* 들어온 인자의 키/요소 값들의 개수를 세서 `dictionary` 형태로 반환
* value 기준으로 내림차순이 선행되고 이 후 key 기준으로 내림차순되어 출력됨 



# collections.Counter()의 연산

#### 1. 덧셈

* `+` 연산 : 공통 key의 value는 서로 더하고 공통 key가 아닌 value는 그대로 value값을 출력

``` python
import collections

a = collections.Counter(['a','b','c'])
b = collections.Counter(['b','c','d'])

print(a+b)		# Counter({'b': 2, 'c': 2, 'a': 1, 'd': 1})
```



#### 2. 뺄셈

* `-` 연산 : 공통 key의 value를 `-` 연산해준다.

* 중요한점으로,  `Counter()` 의  `-` 연산은 음수값을 출력하지 않기 때문에

  공통 key가 아닌 값들은 출력되지 않으며, 공통 key의 차 역시 음수이면 해당 key는 출력되지 않는다.

``` python
import collections

a = collections.Counter(['a','b','c'])
b = collections.Counter(['b','c','d'])

print(a-b)		# Counter({'a': 1})
```



#### 3. 합집합

* 합집합 `|` : 비교 집합 중 key 값의 value가 큰 것들을 몽땅 출력

``` python
import collections

a = collections.Counter('aaabcccde')
b = collections.Counter('abbbcddd')

print(a|b)		Counter({'a': 3, 'b': 3, 'c': 3, 'd': 3, 'e': 1})
```



#### 4. 교집합

* 교집합 `&` : 비교 집합 중 공통인 key값들을 출력하며, 이 때 작은 value값을 묶어서 출력

``` python
import collections

a = collections.Counter('aaabcccde')
b = collections.Counter('abbbcddd')

print(a&b)		Counter({'a': 1, 'b': 1, 'c': 1, 'd': 1})
```



# collections.Counter()의 method

#### 1. `update()`

* `Counter()`의 값을 갱신하는 함수. `Counter()`의 `+` 연산과 유사한 기능

``` python
import collections

a = collections.Counter('aabbcc')
print(a)							# Counter({'a': 2, 'b': 2, 'c': 2})

a.update('abe')
print(a)							# Counter({'a': 3, 'b': 3, 'c': 2, 'e': 1})
```



#### 2. `subtract()`

* `Counter()`의 값을 빼는 함수
* 존재하지 않는 값을 빼려고 하면 음수값이 출력
* `Counter()`의 `-` 연산은 음수값을 출력 못했지만 이 함수를 쓰면 음수가 나올 수 있음

``` python
import collections

a = collections.Counter(['a','b','c'])
b = collections.Counter(['b','c','d'])

a.subtract(b)
print(a)			# Counter({'a': 1, 'b': 0, 'c': 0, 'd': -1})
```



#### 3. `elements()`

* `Counter()`의 key값들을 해당 value 값만큼 출력

``` python
import collections

a = collections.Counter('aaabbc')
a.elements()
print(a)			# Counter({'a': 3, 'b': 2, 'c': 1})
```



#### 4. `most_common([n])`

* `n` 개만큼 가장 빈도가 높은(value 값이 높은) key를 리스트에 담긴 튜플 형태로 출력

``` python
import collections

a = collections.Counter('aaaaabbbbcccdde')
print(a.most_common(3))			# [('a', 5), ('b', 4), ('c', 3)]
```

