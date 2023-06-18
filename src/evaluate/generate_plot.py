from pathlib import Path
from datetime import datetime
from typing import List

import pandas as pd
from matplotlib import pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[2]
if __name__ == '__main__':
    import sys

    sys.path.append(str(PROJECT_DIR))
from submodules.UsefulTools.FileTools.FileOperator import check2create_dir, get_filenames


def get_legend_name(csv_path: str):
    opt = csv_path.split('_')[-3][:-8]
    lr = float(csv_path.split('_')[-3][-8:])
    s = csv_path.split('_')[-2][4]
    try:
        i = int(csv_path.split('/')[-2][-1])
        label = f"{opt}-{lr:.1e}_S{s}"
    except:
        label = f"{opt}-{lr:.1e}_S{s}_handcraft"
    return label


def processing(folder_list: List[str]):
    plotlist = [[], [], [], []]
    # Path = [[train_acc], [train_loss], [val_acc], [val_loss]]
    for folder in folder_list:
        csvList = get_filenames(folder, '*.csv')
        csvList.sort()
        # print(csvList)
        for i, csv in enumerate(csvList):
            plotlist[int(i)].append(csv)
    return plotlist


def plot(plot_list: List[List[str]], start_epoch: int, save_folder: str):
    plt.figure(figsize=(10, 6), dpi=200)
    if len(plot_list[0]) == 1:
        acc_loss = ['acc', 'loss']
        for i in range(2):
            df1 = pd.read_csv(plot_list[i][0])  # train_acc/train_loss
            df2 = pd.read_csv(plot_list[i + 2][0])  # val_acc/val_loss
            columns = df1.columns[1:]
            numberType = acc_loss[i]
            index = min(df1.index[-1], df2.index[-1])
            for label in columns:
                plt.plot(df1.index, df1[label], label='train')
                plt.plot(df2.index, df2[label], label='val')
                plt.title(f"{label}")  # title
                plt.xlabel('epoch')  # x label
                plt.ylabel(f"{numberType}")  # y label
                plt.xlim(start_epoch, index)
                plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize='10', borderaxespad=0)
                # plt.show()
                plt.savefig(save_folder + f"/{numberType}_{label}.png")
                plt.clf()
    else:
        typeList = [['train', 'acc'], ['train', 'loss'], ['val', 'acc'], ['val', 'loss']]
        for i, same_type_csv_list in enumerate(plot_list):
            stepType, numberType = typeList[i]
            df = pd.read_csv(same_type_csv_list[0])
            columns = df.columns[1:]
            for label in columns:
                x_axis_lim = 10000
                for csv in same_type_csv_list:
                    df = pd.read_csv(csv)
                    if len(df) < x_axis_lim:
                        x_axis_lim = len(df)
                    plt.plot(df.index, df[label], label=get_legend_name(csv))
                plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize='8', borderaxespad=0)
                plt.title(f"{label}")  # title
                plt.xlim(start_epoch, x_axis_lim - 1)
                plt.xlabel('epoch')  # x label
                plt.ylabel(f"{numberType}")  # y label
                plt.subplots_adjust(left=0.1, right=0.78, top=0.9, bottom=0.1)
                plt.savefig(save_folder + f"/{stepType}_{numberType}_{label}.png")
                plt.cla()


def generate_plot(path_ls: List[str], output_dir: str, start_epoch: int):
    now = datetime.now()
    time = str(now.strftime("%m%d_%H_%M_%S"))
    des = '/single_model' if len(path_ls) == 1 else '/multi_model'
    check2create_dir(output_dir + des)
    savePath = output_dir + des + '/' + time
    check2create_dir(savePath)
    plotlist = processing(path_ls)
    plot(plotlist, start_epoch, savePath)


if __name__ == "__main__":
    path_ls = ['out/0612-2335_BadmintonNet_BS-15_AdamW1.00e-04_Side2']
    # path_ls = ['out/0612-2335_BadmintonNet_BS-15_AdamW1.00e-04_Side2', 'out/0612-2334_BadmintonNet_BS-15_SGD1.00e-04_Side2','out/0612-2336_BadmintonNet_BS-15_Adam1.00e-04_Side2','out/0612-2334_BadmintonNet_BS-15_RMSprop1.00e-04_Side2']
    generate_plot(path_ls, 'out/plot', 2)
