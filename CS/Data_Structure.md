## Hash Table

* Key와 Value를 1:1로 연관지어 저장하는 자료구조
* Key로 Value 추출 가능

---

#### 기능

* Key, Value가 주어질 때, 두 값을 저장 가능
* Key가 주어졌을 때, 해당 Key에 연관된 Value 조회 가능
* 기존 Key에 새로운 Value가 주어질 때, 기존 Value를 새로운 Value로 대체(1:1)
* Key가 주어졌을 때, 해당 Key에 연관된 Value 제거 가능

---

#### 구성

* Key, Hash Function, Hash, Value, 저장소(Bucket, Slot)로 구성
* Key
  * 고유값
  * 저장 공간의 효율성을 위해 Hash Fucntion에 입력하여 Hash로 변경 후 저장
* Hash Fucntion
  * Key를 Hash로 바꿔주는 역할
  * 해시 충돌(서로 다른 Key가 같은 Hash가 되는 경우)이 발생활 확률을 최대한 줄이는 함수를 만드는 것이 중요
* Hash
  * Hash Function의 결과
  * 저장소에서 Value와 매칭되어 저장
* Value
  * 저장소에 최종적으로 저장되는 값
  * Key와 매칭되어 저장, 삭제, 검색, 접근 가능

---

#### 동작 과정

1. Key -> Hash Function -> Hash(Hash Function의 결과)
2. Hash를 배열의 Index로 사용
3. 해당 Index에 Value 저장
   * HashTable 크기가 10이라면 A라는 Key의 Value를 찾을 때, hashFunction('A') % 10 연산을 통해 인덱스 값을 계산하여 조회

---

#### Hash 충돌

* 서로 다른 Key가 Hash Function에서 중복 Hash로 나오는 경우
* 충돌이 많아질수록 시간 복잡도가 O(1)에서 O(n)으로 증가

#### 해결방법

1. Separating Chaning
   * JDK(자바 개발 키트) 내부에서 사용하는 충돌 처리 방식
   * Linked List(데이터 6개 이하) 또는 Red-Black Tree(데이터 8개 이상) 사용
   * Linked List 사용 시, 충돌이 발생하면 충돌이 발생한 인덱스가 가리키고 있는 Linked List에 노드를 추가하여 Value 삽입
   * Key에 대한 Value 탐색 시에는 인덱스가 가리키고 있는 Linked List를 선형 검색하여 Value 반환(삭제도 같은 방식)
   * Linked List 구조를 사용하기 때문에 추가 데이터 수 제약이 적은 편
2. Open addressing
   * 추가 메모리 공간을 사용하지 않고, HashTable의 빈 공간을 이용하는 방법
   * Separating Chaning 방식에 비해 적은 메모리 사용
   * 방법은 Linear Probing(선형 탐사), Quadratic Probing(제곱 탐사), Double Hashing(이중 해싱)
3. Resizing
   * 저장 공간이 일정 수준 이상 채워지면 Separating Chaning의 경우 성능 향상을 위해, Open addressing의 경우 배열 크기 확장을 위해 Resizing
   * 보통 두배로 확장
   * 확장 임계점은 현재 데이터 개수가 Hash Bucket 개수의 75%가 될 때

---

#### HashTable의 장단점

#### 장점

1. 적은 리소스로 많은 데이터를 효율적으로 관리 가능
   * ex) HDD(하드 디스크 드라이브), Cloud에 있는 많은 데이터를 Hash로 매핑하여 작은 크기의 메모리로 프로세스 관리 가능
2. 배열의 인덱스를 사용하기 때문에 빠른 검색, 삽입, 삭제 가능(O(1))
   * HashTable의 경우, 인덱스는 데이터의 고유 위치이기 때문에 삽입/삭제 시 다른 데이터를 이동할 필요가 없어서 빠른 속도로 삽입/삭제 가능
3. Key와 Hash에 연관성이 없어 보안 유리
4. 데이터 캐싱에 많이 사용
   * get, put 기능에 캐시 로직 추가 시 자주 hit하는 데이터 바로 검색 가능
5. 중복 제거 유용

#### 단점

1. 충돌 발생 가능성
2. 공간 복잡도 증가
3. 순서 무시
4. 해시 함수에 의존

---

#### HashTable vs HashMap

* Key-Value 구조 및 Key에 대한 Hash로 Value를 관리하는 것은 동일
* HashTable
  * 동기 방식
  * null값 허용
  * 보조 Hash Function과 Seperating Chaning을 사용해서 충돌이 비교적 덜 발생
* HashMap
  * 비동기 방식(멀티스레드 환경에서 주의 요망)
  * null갑 미허용(Key가 hashcode(), equals()를 사용하기 때문)

---

#### HashTable 성능

|      | 평균 | 최악 |
| ---- | ---- | ---- |
| 탐색 | O(1) | O(N) |
| 삽입 | O(1) | O(N) |
| 삭제 | O(1) | O(N) |

