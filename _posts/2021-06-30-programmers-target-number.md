---
title: "[Programmers] - DFS/BFS : 타겟넘버 (Level 2)"

categories:
  - Coding test

tags:
  - Coding test
  - Algorithm
  - Python
  - DFS(깊이우선탐색)
  - BFS(너비우선탐색)

layout: single

toc: true
toc_sticky: true
use_math: true
typora-root-url: ../
---

## Programmers Coding Test - 깊이/너비 우선 탐색(DFS/BFS) - Level 2)

안녕하세요, 이번 포스팅은 깊이/너비 우선 탐색과 관련한 문제인데요. 레벨2를 도전해 보았는데 체감난이도가 레벨1에 비해 훨씬 높았습니다 ㅠㅠ. 특히, 연산량이 많아 시간 초과가 되는 것을 방지하기가 참 어려웠습니다. 앞으로 동적할당과 재귀함수는 꾸준히 관심을 가지고 더 공부해야겠다고 다짐했습니다. 그럼, 문제 설명 들어갑니다 !

## 문제설명

n개의 음이 아닌 정수가 있습니다. 이 수를 적절히 더하거나 빼서 타겟 넘버를 만들려고 합니다. 예를 들어 [1, 1, 1, 1, 1]로 숫자 3을 만들려면 다음 다섯 방법을 쓸 수 있습니다.

-1+1+1+1+1 = 3  
+1-1+1+1+1 = 3  
+1+1-1+1+1 = 3  
+1+1+1-1+1 = 3  
+1+1+1+1-1 = 3  

사용할 수 있는 숫자가 담긴 배열 numbers, 타겟 넘버 target이 매개변수로 주어질 때 숫자를 적절히 더하고 빼서 타겟 넘버를 만드는 방법의 수를 return 하도록 solution 함수를 작성해주세요.

### 제한사항
주어지는 숫자의 개수는 2개 이상 20개 이하입니다.
각 숫자는 1 이상 50 이하인 자연수입니다.
타겟 넘버는 1 이상 1000 이하인 자연수입니다.

### 입출력 예

|numbers|target|return|
|---:|:---:|:---|
|[1, 1, 1, 1, 1]|3|5|


## 문제 풀이1

이번 풀이에서 핵심이 되는 부분은 


```python
for items in posneg_itmes:
    result = [x+[y] for x in result for y in items]
```

부분인데요, 모든 경우의 수의 조합을 만들기 위한 코드입니다. 각 요소별로 +1과 -1을 곱한 쌍을 만들어 준 후, 하나씩 추가해주는 방식입니다.


```python
numbers = [1,2,3,4,5]
target = 3
import numpy as np

def solution(numbers, target):
    posneg_items = [(items, -items) for items in numbers]
    
    result = [[]]
    
    for items in posneg_items:
        result = [x+[y] for x in result for y in items]
    
    result = [sum(items) for items in result]
    
    return result.count(target)

solution(numbers,target)
```




    3



## 문제풀이2

다른 사람의 풀이중 재귀함수를 이용해서 아주 아름답게 풀이한 것이 있어서 많은 공부가 되었습니다 :)
앞에서부터 하나씩 줄어드는 number에서 가장 앞에 있는 요소를 각각 target에 더하고 빼면서, target이 0이 되는 부분을 카운팅하는 흐름인데요. 정말 재귀함수를 잘 사용했다는 생각이 들었습니다. 


```python
numbers = [1,2,3,4,5]
target = 3

def solution(numbers, target):
    if not numbers and target == 0 :
        return 1
    elif not numbers:
        return 0
    else:
        return solution(numbers[1:], target-numbers[0]) + solution(numbers[1:], target+numbers[0])
    
solution(numbers, target)
```




    3



이렇게 이번 포스팅도 많은 공부가 되었습니다.  
그럼, 다음 포스팅때 뵙겠습니다.
