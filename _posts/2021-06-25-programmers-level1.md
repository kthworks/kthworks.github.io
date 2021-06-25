---
title: "[Programmers 스킬체크] - Level 1"

categories:
  - Coding test

tags:
  - Coding test
  - Algorithm
  - Python

layout: single

toc: true
toc_sticky: true
use_math: true
typora-root-url: ../
---

## Programmers 스킬체크 Level 1 합격후기

안녕하세요, 오늘은 Programmers의 실력체크 Level1에 도전해 보았습니다!  
스킬 체크는 원하는 언어를 선택해서 제한시간 동안 2문제를 풀어야하는데요.  
모든 테스트 케이스를 통과해야 레벨을 획득할 수 있고, 다양한 테스트 케이스를 통해 코드의 정확성과 효율성을 측정합니다.  
Level1은 제한시간 40분동안 2문제를 푸는 것이고, 통과율은 28.7%, 평균 완료시간은 27.4분으로 나오네요.   
아무래도 코딩을 처음 접하신 분들이 많다보니 시험삼아 들어오셨다가 중도포기하는 경우가 많아 통과율이 낮은 것 같습니다.  

저는 Python으로 Level1을 도전해 보았습니다. 저는 MATLAB에 익숙해 있었던 터라 체감 난이도는 쉬웠고, 머릿속으로 생각한 아이디어를 Python의 표현방식으로 바꾸는 것에 더 집중했던 것 같습니다.  


## 문제 1 - 최대공약수 / 최소공배수

n과 m이 주어졌을 때, n,m의 최대 공약수와 최소 공배수를 출력하는 문제였습니다.

### 문제1 풀이


```python
def solution(n, m):
    a = [] #최대공약수
    b = [] #최소공배수
    
    #최대공약수 구하기
    for i in range(1,n+1):
        if n%i ==0 and m%i==0:
            a=i
    
    #최소공배수 구하기
    t=0
    while True:
        t=t+1
        temp = (n*t)%m
        if temp==0:
            b=n*t
            break
            
    return a,b

solution(n,m)

```




    (1, 12)



## 문제 2 - 짝수 / 홀수
주어진 숫자 n이 짝수면 "Even", 홀수면 "Odd"를 출력하는 아주 기초적인 문제입니다.
% 연산자를 이용하여 쉽게 풀었습니다.

### 문제 2 풀이


```python
def solution(num):
    if num%2 ==0:    
        answer = "Even"
    else:
        answer = "Odd"
    return answer

print(solution(10))
print(solution(5))
```

    Even
    Odd
    

이렇게 저는 스킬 체크 레벨1을 합격했습니다.  
다음 레벨들도 쭉쭉 합격해나가도록 하겠습니다.

그럼, 다음 포스팅때 또 뵙겠습니다 :)