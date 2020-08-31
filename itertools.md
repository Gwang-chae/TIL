# itertools

* `itertools` 패키지
* 간결하고 메모리 효율적인 도구 제공



## Infinite iterators

* #### count()

  ```python
  count(start, [step])
  # 출력 예
  count(10)
  >>> 10 11 12 13 14 15 ...
  ```

  * start 부터 [step] 만큼 더해간 결과값들을 무한 출력
  * start, start+step, start+2*step, ...

* #### cycle()

  ```python
  cycle(p)
  # 출력 예
  cycle('ABCD')
  >>> A B C D A B C D A B ...
  ```

  * 인자로 넣은 값의 첫 인자부터 끝 인자까지 차례도록 무한 출력
  * p[0], p[1], ... p[last], p[0], p[1], ...

* #### repeat()

  ``` python
  repeat(elem, [n])
  # 출력 예
  repeat(10, 5)
  >>> 10 10 10 10 10
  ```

  * elem를 n번 반복
  * 이 때 n은 옵션으로 n값을 주지 않으면 무한 출력
  *  elem, elem, elem, ... 끝없이 또는 최대 n 번

