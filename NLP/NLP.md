# NLP

`자연어 처리(Natural Language Processing)`

* 자연어의 의미를 분석하여 컴퓨터로 다양한 문제를 해결하는 것

<br>

---

## 자연어 처리의 종류

1. 텍스트 분류(Text Classification)
   - 특정 문장이나 문서를 어떠한 카테고리로 분류하는 문제
   - ex) 스팸 메일 분류
2. 감성 분석(Sentimental Analysis)
   - 특정 문장이나 문서의 긍/부정을 파악
   - ex) 챗봇이 상대의 감정을 파악하여 대화 주제를 바꾸는 경우
3. 내용 요약(Text Summarization)
   - 추출 요약과 생성 요약으로 구분
   - 1. 추출 요약 : 문서에서 중요하다고 여겨지는 문장들을 뽑아 요약문으로 이용하는 방법
     2. 생성 요약 : 요약문을 새롭게 생성하는 방법
4. 기계 번역(Machine Translation)
   - 서로 다른 나라의 언어로 번역해주는 무제
5. 챗봇(Chat Bot)
   - 일상에서 가장 깊숙히 자리잡은 자연어 처리 분야
   - 다양한 어플리케이션에서 내부적으로 구현

<br>

---

## 자연어 처리 과정

* 크게 아래와 같은 순서로 자연어 처리가 이루어짐

1. Preprocessing
2. Vectorization
3. Embedding
4. Modeling

<br>

---

## Preprocessing

* 모델의 입력인 단어, 문장, 문서의 vector를 만들기 전까지 진행되는 과정
* 말뭉치(`Corpus`)로부터 문법적으로 더 이상 나눌 수 없는 언어요소 토큰(`Token`)으로 나누는 작업을 `Tokenization`
* 대표적인 자연어 처리 패키지로 `NLTK`
* 한국어 처리 패키지로 `KoNLPy`가 존재하며, `KoNLPy`의 `Kkma()`, `Okt()`, `Mecab()`등이 사용됨 

1. 어간 추출 or 형태소 분석(`Stemming`)
   * 단어나 문장의 언어적 속성을 파악하는 단계
   * 주어진 단어에서 핵심 의미를 담고 있는 부분을 찾는 과정
   * 단어를 의미를 담고 있는 어간과 문법적 역할을 하는 접사로 분리하는 동작 방식
2. 표제어 추출(`Lemmatisation`)
   * 주어진 단어의 사전적 어원을 찾는 과정
   * 단어와 단어의 **품사**를 바탕으로 단어사전과 비교하여 표제어를 찾는 방식으로 동장
   * 품사(`Part Of Speech`)로 판단
3. 불용어 제거(`Stopwords removing`)
   * 문장에서 큰 의미가 없다고 생각하는 단어를 불용어라 지칭
   * 불용어는 문장에서 자주 등장하지만 실제 의미 분석에는 사용되지 않으므로 미리 불용어를 정의하고 제거

<br>

---

## Vectorization

* 자연어를 컴퓨터가 이해할 수 있게 숫자로 바꾸는 작업
* 이 때 벡터로 변환된 고유의 토큰들이 모인 집합을 `vocabulary`라 하며, `vocabulary`가 클수록 학습 시간은 오래 걸림
* `vectorization` 기법
  1. One Hot Encoding
     - 단어가 존재하면 1, 그렇지 않으면 0으로 표시하는 기법
     - 단어의 수가 많아지면 `vocabulary`의 크기가 커짐
     - 의미적으로 유사한 단어 관계를 파악하기 어려움
  2. Bag of words 
     * `vocabulary`를 활용하여 각 문장이 갖고 있는 토큰의 **빈도**를 기반으로 문장을 vectorization
     * 단어의 수가 많아질수록 `vocabulary`의 크기가 커짐
     * 단어간의 의미 관계 파악이 어려움
  3. TF-IDF(Term Frequency - Inverse Document Frequency)
     * TF(Term Frequency) : 특정 단어가 문서 내에 얼마나 자주 등장하는지를 나타내는 값
     * TF가 높을수록 문서에서 중요하다고 할 수 있음, 그러나 문서군 내에서 자주 사용될 경우 해당 단어는 흔하게 등장한다는 것을 의미
     * DF(Document Frequency) : 문서군 내에서 얼마나 자주 등장하는지를 나타내는 값
     * IDF(Inverse Document Frequency) : DF의 역수
     * TF-IDF는 TF와 IDF의 곱으로 점수가 높은 단어일수록 다른 문서에는 많지 않고 해당 문서에서 자주 등장하는 단어

* 기본적으로는 vectorization한 후 문장들의 길이를 맞춰줘야 하기 때문에 Padding 작업 실시

<br>

---

## Embedding

* Vectorization은 단어의 중요도, 문서 안에서의 중요도는 표현가능하지만 단어 사이의 유사도, 관계는 설명하지 못하는 단점이 존재
* 비슷한 의미를 지닌 토큰들끼리는 서로 가깝게, 그렇지 않은 토큰들은 멀리 배치하는 것이 Embedding의 목적
* Embedding도 하나의 모델을 훈련하는 것을 의미하며, 보통은 pre-trained model을 사용

* `Embedding` 방법
  1. Keras Embedding Layer
     * 가장 쉽고 빠르게 네트워크 모델에 Embedding 층을 주입하는 방식
     * 무작위로 특정 차원으로 입력 벡터들을 뿌린 후, 학습을 통해 가중치를 조정해 나가는 방법
     * **단어 사이의 관계를 반영하는 방법은 아님**
  2. word2vec
     * 비슷한 위치에서 등장하는 단어들은 비슷한 의미를 가진다는 가정하에 고차원의 `sparse`한 벡터를 상대적으로 저차원의 형태로 분산시켜 단어의 유사도를 계산
     * `word2vec`의 방식
       1. CBOW(Continous Bag of Words)
          - 주변 단어들을 가지고, 중간에 있는 단어들을 예측하는 기법
          - window size만큼의 앞뒤 주변 단어로 중심 단어를 예측
       2. Skip-gram
          * 중심 단어에서 주변 단어를 예측
          * 전반적으로 Skip-gram이 CBOW보다 성능이 좋음
  3. glove
     * word2vec은 사용자가 지정한 주변 단어의 개수에 대해서만 학습이 이루어지기 때문에 데이터 전체에 대한 정보를 담기 어려움
     * 이를 보완하고자 glove는 토큰들 간의 유사성은 그대로 가져가면서 데이터 전체에 대한 빈도를 반영
  4. FastText
     * 페이스북에서 개발한 Embedding 방법
     * word2vec은 단어를 쪼개질 수 없는 단위로 생각하는 반면, FastText는 하나의 단어 안에도 여러 sub 단어들이 존재하는 것으로 간주하여 sub 단어들을 단위로 사용

<br>

---

## Modeling

* 자연어 처리에서는 ML보다 네트워크 모델을 이용하여 문제를 해결하는 것이 일반적
* Modeling 종류
  1. RNN(Recurrent Neural Network)
     * 자연어, 음성신호와 같은 연속적인, 시계열 데이터에 적합한 모델
     * 입력층 -> 출력층의 한 방향으로만 흐르는 `feedforward`와 비슷하지만, 출력을 다시 입력으로 받는 부분이 존재
     * 타임 스텝(`t`)마다 모든 뉴런은 입력 벡터와 이전 타임 스텝(`t-1`)의 출력 벡터를 입력으로 받음
     * 타임 스텝이 길어질수록 앞쪽의 타임스텝은 영향을 주지 못하는 **장기 의존성**(`Long-Term Dependency`) 문제 발생
     * 순차적인 구조로 인해 연산에 많은 시간이 소요
     * 기울기 소실 문제 존재
  2. LSTM(Long Short-Term Memory)
     * 기존 RNN에 `cell state`를 추가한 모델
     * `cell state`는 입력정보를 선별하여 다음 출력으로 내보내는 `gate`역할 수행
     * 불필요한 정보를 걸러내어 매끄러운 진행이 가능하고 기울기 소실 문제를 줄여 성능을 증가시킴
  3. GRU(Gated Recurrent Unit)
     * LSTM의 간소화된 버전
     * LSTM의 장점을 가져오면서 속도적인 부분을 개선하여 빠른 속도로 비슷한 성능 발휘
  4. seq2seq(Sequence to Sequence)
     * RNN과 같은 모델들은 입력 시퀀스를 통해 하나의 출력을 하거나 각 입력으로부터 각 출력을 생성
     * 또한, 입력과 출력 시퀀스의 크기는 고정되어 있었음
     * 한 언어를 다른 언어로 번역하는 기계 번역 문제에서는 문장 길이가 달라질 수 있기 때문에 출력 시퀀스에 대한 처리가 필요
     * 각 시퀀스에 대한 길이를 미리 알 수 없고, 입력 시퀀스를 출력 시퀀스로 생성해야하는 문제를 해결하기 위해 제안된 모델이 seq2seq
     * 입력 시퀀스를 처리하는 Encoder, 출력 시퀀스를 생성하는 Decoder로 구분
     * Encoder 부분에서는 LSTM(RNN계열)층을 통해 hidden state를 넘겨주면서 마지막 단계의 hidden state(`context vector`)를 Decoder에 전달
     * Decoder에서는 전달받은 `context vector`를 통해 다음 나올 단어를 예측하고, 그 예측된 단어가 다시 다음 스텝의 입력으로 들어가 계속해서 예측하는 방식
     * seq2seq에서는 교사강요(`Teacher Forcing`) 학습방법을 적용
       * `Teacher Forcing` : 학습할 때 예측값이 input이 아닌 실제값을 input으로 적용하여 학습하는 방법
  5. Attention
     * seq2seq 모델은 하나의 고정된 크기의 벡터에 모든 정보를 압축하려다 보니 정보손실이 발생하고, 기울기 소실 문제 또한 존재
     * 입력 시퀀스가 길어지면 출력 시퀀스의 정확도가 떨어지는 것을 보정해주기 위해 등장한 기법이 Attention
     * Decoder에서 출력 단어를 예측하는 매 시점마다 Encoder에서의 전체 입력 문장을 다시 한 번 참고
     * 전체 입력 문장을 모두 동일한 비율로 참고하는 것이 아닌, 해당 시점에서 예측해야할 단어와 연관 있는 입력 단어를 더 집중해서 참고

