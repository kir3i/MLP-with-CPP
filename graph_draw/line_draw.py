import matplotlib.pyplot as plt
import numpy as np

# 점 색깔 설정
color = ['red', 'green', 'orangered', 'm', 'k', 'cyan', 'lime', 'lightgrey', 'brown']

# 직선 하나를 그리기
def draw_line(line, dot, name, epoch, layerNum, nodeNum):
    # epoch, layerNum, nodeNum은 1부터 시작
    epoch += 1
    layerNum += 1
    nodeNum += 1

    # line 정보 해석
    m, n = map(float, line.split(' '))

    # dot 정보 해석
    dot = [tuple(map(float, d.split(' '))) for d in dot]

    # 점 찍기
    for (a, b), c in zip(dot, color):
        plt.scatter(a, b, label=f'({a:.2f}, {b:.2f})', color=c)

    # 직선그래프 그리기 위한 x, y 배열 준비
    x = np.arange(-0.1, 1.1, 0.1)
    y = [m*k+n for k in x]

    # 그래프 그리기
    plt.plot(x, y, label=f'x2={m:.3f}*x1 {"+" if n>=0 else "-"} {abs(n):.3f}')
    # 제목
    plt.title(f'{name}, {layerNum}-{nodeNum}, epoch: {epoch}')
    # 축 이름
    plt.xlabel('x1')
    plt.ylabel('x2')
    # 눈금 표시
    plt.grid()
    # 범례 표시
    if name == 'DONUT':
        plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    else:
        plt.legend()

    # 그래프를 파일로 저장
    if name == 'DONUT':
        plt.savefig(f'{name}_{epoch}_{layerNum}_{nodeNum}.png', bbox_inches='tight')
    else:
        plt.savefig(f'{name}_{epoch}_{layerNum}_{nodeNum}.png')
    # 그래프 영역 초기화
    plt.cla()

# layer 하나를 그리기
def draw_layer(layer, dot, name, epoch, layerNum):
    # layer단위로 쪼개기
    layer = layer.split('\n')

    # 기본 점 설정
    if dot == None and name != 'DONUT':
        dot = ['0 0', '0 1', '1 0', '1 1']
    elif dot == None:
        dot = ['0 0', '0 1', '1 0', '1 1', '0.5 1', '1 0.5', '0 0.5', '0.5 0', '0.5 0.5']

    # 각 node에 대해 직선 그리기
    for nodeNum, line in enumerate(layer):
        draw_line(line, dot, name, epoch, layerNum, nodeNum)

if __name__ == '__main__':
    # 출력할 테스트셋 설정
    filename = 'XOR'

    # 눈금 간격 설정
    INTERVAL = 100

    with open(f'{filename}_line.txt', 'r') as f, open(f'{filename}_dot.txt', 'r') as dts:
        # epochs 추출
        epochs = f.read().split(',')[:-1]
        dots = dts.read().split('\n\n')[:-1]

        # 출력할 epoch 선택
        print_epoch = [29, 59, 149]
        # print_epoch = [x for x in range(INTERVAL-1, len(epochs), INTERVAL)]
        # print_epoch.append(0)
        # print_epoch.append(len(epochs)-1)

        # 각 epoch마다 실행 결과 추출
        for epoch, (layers, dot) in enumerate(zip(epochs, dots)):
            # 선택한 epoch만 출력
            if epoch not in print_epoch:
                continue

            # layers 정보 해석
            layers = layers.split('\n\n')[:-1]
            # (이동한) 점 정보
            dot = dot.split('\n')

            for layerNum, layer in enumerate(layers):
                if layerNum == 0:
                    draw_layer(layer, None, filename, epoch, layerNum)
                elif layerNum == 1:
                    draw_layer(layer, dot, filename, epoch, layerNum)
