## Git 기초

> Git은 분산형 버전관리시스템(DVCS)이다.

Git을 윈도우에서 활용하기 위해서는 [git bash](https://gitforwindows.org/)를 설치해야 한다.

## 1. 저장소 초기화

```bash
$ git init
Initialized empty Git repository in C:/Users/i/Desktop/TIL/.git/

(master) $
```

* 로컬 저장소를 만들고  나면, `.git/` 폴더가 생성되고, bash에 `(master)`라고 표기 된다.

* 반드시 저장소를 만들기 전에 원하는 디렉토리인지 

  확인하는 습관을 가지고, 저장소 내부에 저장소를 만들지 말자
  * ex) Desktop -> git 저장소, 

    ​	   Desktop/TIL -> 다른 git 저장소가 존재하면 안됨

## 2. add

작업한 내용을 커밋 대상 목록에 추가한다.

```bash
$ git add .				# 현재 디렉토리(하위 디렉토리 포함)
$ git add a.html		# 특정 파일
$ git add b.html c.html	# 특정 다수 파일
$ git add blog/			# 특정 폴더
```



```bash
# 작업 후 상태
$ git status
On branch master

No commits yet
# Untracked files => Git으로 관리된 적 없는 파일
Untracked files:
# 커밋될 것들에 포함시려면 add 명령어를 써라
  (use "git add <file>..." to include in what will be committed)
        markdown-images/    "\353\247\210\355\201\254\353\213\244\354\232\264 \353\254\270\353\262\225.md"

# 총평
# 커밋될 것이 없음
# 하지만, 새로 생성한 파일(Untracked files은 존재함)
nothing added to commit but untracked files present (use "git add" to track)
```

```bash
$ git add .
```

```bash
# add 명령어 입력 후 상태
$ git status
On branch master

No commits yet
# 커밋이 될 변경사항들
# working directory X
# Staging area O
Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
        new file:   markdown-images/typora_setting-1597900376898.PNG
        new file:   markdown-images/typora_setting.PNG
        new file:   "\353\247\210\355\201\254\353\213\244\354\232\264 \353\254\270\353\262\225.md"

```

## 3. 커밋

```bash
$ git commit -m 'Add markdown.md'
[master (root-commit) a64db28] Add markdown.md
 3 files changed, 84 insertions(+)
 create mode 100644 markdown-images/typora_setting-1597900376898.PNG
 create mode 100644 markdown-images/typora_setting.PNG
 create mode 100644 "\353\247\210\355\201\254\353\213\244\354\232\264 \353\254\270\353\262\225.md"
```

* 커밋은 버전(이력)을 기록하는 명령어이다.

* 커밋 메세지는 해당하는 이력을 나타낼 수 있도록 

  작성 하여야 한다.

* 커밋 이력을 확인하기 위해서는 아래의 명령어를 사용한다.

``` bash
$ git log
commit a64db2827bfe8847a972e4865ef7946e4fda0710 (HEAD -> master)
Author: Gwang-chae <kcseo25@naver.com>
Date:   Thu Aug 20 14:58:31 2020 +0900

    Add markdown.md

$ git log -1
$ git log --oneline
$ git log --oneline -1
```

