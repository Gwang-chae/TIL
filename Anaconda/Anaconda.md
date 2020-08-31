# Anaconda

* 데이터 분석쪽은 개발IDE로 `jupyter notebook`을,
* 개발환경으로는 `Anaconda` 가상환경을 만들어서 사용



## Anaconda 설치

* `Anaconda` (open source version) 설치
  * `Anaconda`는 오픈 소스 
  * `python`과 다수의 유용한 package를 사용하기 쉽도록 제공해 주는 플랫폼



## Anaconda 환경설정

* `Anaconda`의 Prompt를 관리자 권한으로 실행
* 최신 버전의 pip로 업그레이드

```
python -m pip install --upgrade pip
```

* `data_env`라는 이름의 가상 환경 생성

``` 
conda create -n data_env python=3.7 openssl
```

* 정상적으로 가상환경이 만들어졌는지 확인

``` 
conda info --envs
```

* 가상환경 `data_env`로 전환

``` 
activate data_env
```

* 현재 가상환경에서 추후 `tensorflow` 사용을 위해 필요 package 설치

``` 
conda install nb_conda
```

* jupyter notebook이 사용할 기본 디렉토리 설정
  * 이 작업을 위해 환경설정 파일을 생성이 선행되어야 함

```
jupyter notebook --generate-config
```

* 만들어진 config file에서 디렉토리 경로 수정 (`'C:/notebook_dir'`)

``` 
c.NotebookApp.notebook_dir = 'C:/notebook_dir'
```



## IDE 실행

* 가상환경에서 IDE 실행

``` 
jupyter notebook
```

* `jupyter notebook`이 실행되면 `new` 탭에서 가상환경이 올바르게 설정됐는지 확인