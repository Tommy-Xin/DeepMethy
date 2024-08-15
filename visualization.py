import matplotlib.pyplot as plt
import pandas as pd
def plot_training_history(record_file):
    # 获取训练过程中的损失率和准确率  record_file格式为 epoch loss accuracy

    f = pd.read_csv(record_file, header=0)
    loss = f['loss']
    accuracy = f['accuracy']

    # 获取训练周期数
    epochs = [i + 1 for i in range(len(loss))]

    # 绘制损失率图像
    # plt.plot(epochs, loss, 'b', label='Training loss')
    # plt.plot(epochs, accuracy, 'r', label='Training accuracy')
    #
    # plt.title('Training Metrics')
    # plt.xlabel('Epochs')
    # plt.ylabel('Metrics')
    # plt.legend()
    # plt.show()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 绘制准确率图像
    plt.plot(epochs, accuracy, 'r', label='Training accuracy')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


plot_training_history("dataset/training_metrics_win(41,25,11)_Code(one-hot).csv")