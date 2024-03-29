---
title: "[Programmers] - 크레인 인형뽑기 (Level 1)"

categories:
  - kakao

tags:

  - coding_test


layout: single

toc: true
toc_sticky: true
use_math: true
typora-root-url: ../
comments: true
---

## Programmers Coding Test 2019 카카오 개발자 겨울 인턴십

안녕하세요, 이번 포스팅은 **2019 카카오 개발자 겨울 인턴십 기출문제**입니다 ㅎㅎ  
크레인 인형뽑기 게임을 기반으로 출제된 문제인데요! 바로 문제 설명으로 들어가겠습니다.

### 문제 설명

게임개발자인 "죠르디"는 크레인 인형뽑기 기계를 모바일 게임으로 만들려고 합니다.  
"죠르디"는 게임의 재미를 높이기 위해 화면 구성과 규칙을 다음과 같이 게임 로직에 반영하려고 합니다.  

![](/images/puppet_draw/crane_game_101.png)  

게임 화면은 "1 x 1" 크기의 칸들로 이루어진 "N x N" 크기의 정사각 격자이며 위쪽에는 크레인이 있고 오른쪽에는 바구니가 있습니다. (위 그림은 "5 x 5" 크기의 예시입니다). 각 격자 칸에는 다양한 인형이 들어 있으며 인형이 없는 칸은 빈칸입니다. 모든 인형은 "1 x 1" 크기의 격자 한 칸을 차지하며 격자의 가장 아래 칸부터 차곡차곡 쌓여 있습니다. 게임 사용자는 크레인을 좌우로 움직여서 멈춘 위치에서 가장 위에 있는 인형을 집어 올릴 수 있습니다. 집어 올린 인형은 바구니에 쌓이게 되는 데, 이때 바구니의 가장 아래 칸부터 인형이 순서대로 쌓이게 됩니다. 다음 그림은 [1번, 5번, 3번] 위치에서 순서대로 인형을 집어 올려 바구니에 담은 모습입니다.  


![](/images/puppet_draw/crane_game_102.png)

만약 같은 모양의 인형 두 개가 바구니에 연속해서 쌓이게 되면 두 인형은 터뜨려지면서 바구니에서 사라지게 됩니다. 위 상태에서 이어서 [5번] 위치에서 인형을 집어 바구니에 쌓으면 같은 모양 인형 두 개가 없어집니다.  

![](/images/puppet_draw/crane_game_103.png)

크레인 작동 시 인형이 집어지지 않는 경우는 없으나 만약 인형이 없는 곳에서 크레인을 작동시키는 경우에는 아무런 일도 일어나지 않습니다. 또한 바구니는 모든 인형이 들어갈 수 있을 만큼 충분히 크다고 가정합니다. (그림에서는 화면표시 제약으로 5칸만으로 표현하였음)

게임 화면의 격자의 상태가 담긴 2차원 배열 board와 인형을 집기 위해 크레인을 작동시킨 위치가 담긴 배열 moves가 매개변수로 주어질 때, 크레인을 모두 작동시킨 후 터트려져 사라진 인형의 개수를 return 하도록 solution 함수를 완성해주세요.

#### 제한사항

- board 배열은 2차원 배열로 크기는 "5 x 5" 이상 "30 x 30" 이하입니다.
- board의 각 칸에는 0 이상 100 이하인 정수가 담겨있습니다.
- 0은 빈 칸을 나타냅니다.
- 1 ~ 100의 각 숫자는 각기 다른 인형의 모양을 의미하며 같은 숫자는 같은 모양의 인형을 나타냅니다.
- moves 배열의 크기는 1 이상 1,000 이하입니다.
- moves 배열 각 원소들의 값은 1 이상이며 board 배열의 가로 크기 이하인 자연수입니다.

#### 입출력 예  

|board|moves|result|
|:---:|:---:|:---:|
|[[0,0,0,0,0],[0,0,1,0,3],[0,2,5,0,1],[4,2,4,4,2],[3,5,1,3,1]]|[1,5,3,5,1,2,1,4]|4|

#### 입출력 예에 대한 설명

###### 입출력 예 #1

인형의 처음 상태는 문제에 주어진 예시와 같습니다. 크레인이 [1, 5, 3, 5, 1, 2, 1, 4] 번 위치에서 차례대로 인형을 집어서 바구니에 옮겨 담은 후, 상태는 아래 그림과 같으며 바구니에 담는 과정에서 터트려져 사라진 인형은 4개 입니다.  

![](/images/puppet_draw/crane_game_104.png)

### 문제 풀이


```python
import numpy as np

board = [[0,0,0,0,0],[0,0,1,0,3],[0,2,5,0,1],[4,2,4,4,2],[3,5,1,3,1]]
moves = [1,5,3,5,1,2,1,4]

def solution(board, moves):

    bag = []  
    count = 0
    tr_board = np.transpose(board)

    for i in range(0,len(moves)):
        if (tr_board[moves[i]-1].sum() != 0):
            temp = tr_board[moves[i]-1]
            ind = np.nonzero(temp)
            bag.append(temp[ind[0][0]])
            tr_board[moves[i]-1][ind[0][0]] = 0

            if len(bag)>1 and bag[-2]==bag[-1]:
                count += 2
                bag.pop()
                bag.pop()

    return count

solution(board, moves)




```




    4



이번 풀이는 효율성보다는 직관적으로 풀려고 했던 것 같습니다. 크레인은 column을 따라서 움직이기 때문에 board 데이터를 transpose해서 처리하기 쉽게 바꿨습니다. 다음으로, moves에서 받은 column index를 통해서 가장 위에 있는 인형 위치를 nonzero 함수를 통해 찾아주었습니다. 마지막으로, 각 column index에 따라 인형을 하나씩 뽑은 후 bag에 넣고, 같은 인형이 있으면 pop으로 제거해준 후 count를 해주는 방식으로 풀었습니다. 문제의 흐름에 충실하게 풀었던 것 같습니다 ㅎㅎ  

그럼, 저는 다음 포스팅때 또 뵙겠습니다 :)
