import matplotlib.pyplot as plt
import numpy as np

# 그래프 눈금 간격 설정
INTERVAL = 100

# loss 그래프 그리기
def drawLoss(name, loss):
    # epoch
    x = np.array(range(1, len(loss)+1, 1))

    # 그래프 그리기
    plt.plot(x, loss, label='loss', color='orangered')
    # 눈금 표시
    plt.grid()
    # x축 값 표시 설정
    xticks = [1]
    xticks += list(range(INTERVAL, len(x)+1, INTERVAL))
    plt.xticks(xticks, rotation=45)
    # y축 범위 설정
    plt.ylim((0, max(loss)+0.05*max(loss)))
    # 축 이름 설정
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # 제목 설정
    plt.title(f'{name} loss')
    # 범례 표시
    plt.legend()

    # 그래프 파일로 저장
    plt.savefig(f'{name}_loss.png', bbox_inches='tight')
    # 그래프 초기화
    plt.cla()

if __name__ == '__main__':
    filenames = ['AND', 'OR', 'XOR', 'DONUT']

    # 긱 파일에 대하여 loss 그래프 그리기
    for filename in filenames:
        with open(f'{filename}_loss.txt') as f:
            loss = list(map(float, f.read().split('\n')[:-1]))
            INTERVAL = int(len(loss) / 10)
            drawLoss(filename, loss)
        