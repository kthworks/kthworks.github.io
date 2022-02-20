---
title: "[Programmers] - Hash : 완주하지 못한 선수 (Level 1)"

categories:
  - basic

tags:

  - Algorithm
  - Hash

layout: single

toc: true
toc_sticky: true
use_math: true
typora-root-url: ../
comments: true
---

## Programmers Coding Test 연습시리즈

안녕하세요, 이번 포스팅은 'Hash (해시)'를 적극적으로 이용하여 푸는 문제입니다.  
레벨은 1이었지만, 해시를 잘 사용하지 않으면 효율성을 만족하지 못해 꽤 고생을 했네요.  

Python에서는 Dictionary가 Hash의 대표적인 예인데요, 저도 이번 기회에 Hash에 대해서 공부를 하면서 코드를 짜봤습니다.

### 문제 설명

수많은 마라톤 선수들이 마라톤에 참여하였습니다. 단 한 명의 선수를 제외하고는 모든 선수가 마라톤을 완주하였습니다.  

마라톤에 참여한 선수들의 이름이 담긴 배열 participant와 완주한 선수들의 이름이 담긴 배열 completion이 주어질 때, 완주하지 못한 선수의 이름을 return 하도록 solution 함수를 작성해주세요.

### 제한사항
마라톤 경기에 참여한 선수의 수는 1명 이상 100,000명 이하입니다.  
completion의 길이는 participant의 길이보다 1 작습니다.  
참가자의 이름은 1개 이상 20개 이하의 알파벳 소문자로 이루어져 있습니다.  
참가자 중에는 동명이인이 있을 수 있습니다.  

### 입출력 예


|participant|completion|return|
|:---:|:---:|:---:|
|["leo", "kiki", "eden"]|["eden", "kiki"]|"leo"|
|["marina", "josipa", "nikola", "vinko", "filipa"]|["josipa", "filipa", "marina", "nikola"]|"vinko"|
|["mislav", "stanko", "mislav", "ana"]|["stanko", "ana", "mislav"]|"mislav"|

### 입출력 예 설명

#### 예제 #1
"leo"는 참여자 명단에는 있지만, 완주자 명단에는 없기 때문에 완주하지 못했습니다.  

#### 예제 #2
"vinko"는 참여자 명단에는 있지만, 완주자 명단에는 없기 때문에 완주하지 못했습니다.  

#### 예제 #3
"mislav"는 참여자 명단에는 두 명이 있지만, 완주자 명단에는 한 명밖에 없기 때문에 한명은 완주하지 못했습니다.


## 문제 풀이


```python
participant = ["marina", "josipa", "nikola", "vinko", "filipa"]
completion = ["josipa", "filipa", "marina", "nikola"]

def solution(participant, completion):

    par = {}
    for i in range(0,len(participant)):
        try: par[participant[i]] +=1
        except: par[participant[i]] =1

    for i in range(0,len(completion)):
        try: par[completion[i]] -=1
        except: par[compltion[i]] =0

    key = list(par.keys())
    val = list(par.values())

    return key[val.index(1)]

solution(participant, completion)

```




    'vinko'



## 다른 풀이

다른분들이 풀어주신 풀이법도 살펴 봤는데요, 그 중 hash를 정말 제대로 이용하여 기발하게 푼 풀이가 있어 공유해봅니다.  
주 핵심은, Hash의 숫자값의 가감을 이용한 것이 참 인상적이었습니다.


```python
answer = ''
temp = 0
dic = {}
for part in participant:
    dic[hash(part)] = part
    temp += int(hash(part))

for com in completion:
    temp -= hash(com)
answer = dic[temp]

print(answer)

```

    vinko


이렇게 문제를 하나하나 풀다 보면, 제 부족한점을 많이 채울 수 있는 것 같아 기분이 좋습니다.  
다음에는 이번에 배운 해쉬를 적극적으로 활용하여 문제를 풀어보면 좋을 것 같네요.  

그럼, 다음 포스팅때 또 뵙겠습니다 :)
