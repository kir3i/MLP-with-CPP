# MLP-with-cpp

3학년 2학기 인공지능 과목 과제로 수행한 결과물입니다.

## C++로 MLP를 바닥부터 구현하기
과제의 요구사항은 다음과 같습니다.
- XOR 게이트를 학습할 수 있는 MLP 만들기
- DONUT 게이트를 학습할 수 있는 MLP 만들기

각 gate의 입력 차원은 2이고 출력 차원은 1입니다.

- 구현 환경
  - MS Visual Studio Community 2017
  - C++17
- 컴파일 환경
  - MSVC++ 14.16, x86
 
 ## Multi Layer Perceptron
실제 이 프로젝트로 구현한 MLP 예시입니다.
![image](https://user-images.githubusercontent.com/44166353/108863766-e5c1ea80-7634-11eb-8d95-149b5a2781c6.png)

- 하나의 Perceptron은 선형 연산을 하는 unit입니다.
- 한 Layer는 여러 개의 Perceptron으로 이루어집니다.
- 여러 Layer를 쌓아서 하나의 Network를 만드는데, 이것이 MLP입니다.

 ## 주요 클래스
- Perceptron
  - 가장 기초가 되는 unit인 Perceptron을 구현한 클래스입니다.
  - 입력 차원 `n`과 learning rate `lr`을 초기 설정할 수 있습니다.
  - forward 연산과 backward 연산(`update_weight`)이 존재합니다.
    - forward 연산에서는 `w*x+b`를 계산하여 다음 결과를 냅니다.
    - backward 연산에서는 delta값을 토대로 기울기를 구합니다.
- Layer
  - 여러 개의 perceptron이 모여서 이룬 layer를 구현한 클래스입니다.
  - 입력 차원 `input_dim`과 출력 차원 `output_dim`, 그리고 learning rate `lr`을 초기 설정할 수 있습니다.
    - 이 때 입력차원은 해당 Layer가 가진 Perceptron의 입력차원과 같고, 출력 차원은 해당 Layer가 가진 Perceptron의 수와 같습니다.
  - 각 Layer는 Perceptron을 vector로 저장하고 있습니다.
  - forward 연산과 backward 연산이 존재합니다.
    - forward 연산에서는 각 perceptron이 forward 연산을 수행하도록 합니다.
    - backward 연산에서는 각 perceptron이 backward 연산을 수행하도록 합니다. 이 때 뒷레이어에서 넘어온 `delta_bar`값을 각 perceptron에게 넘겨줘야 합니다.
- Model
  - 여러 개의 layer가 모여서 이룬 network를 구현한 클래스입니다.
  - 입력 차원 `input_dim`과 network를 이루는 Layer들의 차원 정보를 담고 있는 `layers_dim`, learning rate `lr`을 초기 설정할 수 있습니다.
  - run 연산이 존재하며 여기서 각 Layer의 forward 연산, backward 연산을 수행합니다.
 
 ## 실행 및 테스트 방법
이 프로젝트의 모든 코드는 Visual Studio Community 2017로 작성되었으므로 코드 다운로드 후 Visual Studio에서 실행하여 테스트할 수 있습니다.
