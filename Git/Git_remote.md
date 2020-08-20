## Git 원격 저장소 활용

Git 원격 저장소 기능을 제공해주는 서비스는 `gitlab`, `bitbucket`, `github` 등이 있다.

## 0. 원격 저장소 설정

```bash
$ git remote add origin {url}
# git remote add origin https://github.com/Gwang-chae/TIL.git
```

* git 원격저장소를 추가(`add`)하고 `origin`이라는 이름으로 `url`으로 설정
* 설정된 저장소를 확인하기 위해서는 아래의 명령어를 사용한다.

```bash
$ git remote -v
origin  https://github.com/Gwang-chae/TIL.git (fetch)
origin  https://github.com/Gwang-chae/TIL.git (push)
```



## 1. 원격 저장소에 `push`

```bash
$git push origin master
Enumerating objects: 10, done.
Counting objects: 100% (10/10), done.
Delta compression using up to 8 threads
Compressing objects: 100% (10/10), done.
Writing objects: 100% (10/10), 27.22 KiB | 9.07 MiB/s, done.
Total 10 (delta 0), reused 0 (delta 0), pack-reused 0
To https://github.com/Gwang-chae/TIL.git
 * [new branch]      master -> master
Branch 'master' set up to track remote branch 'master' from 'origin'.
```