import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "H:\SUA\SUA\\Organized Files\\sub_001.csv"
print(path[-11:-4])
trip_spd_count = 0
fig_spd_count = 1
trip_acc_count = 0
fig_acc_count = 1
trip_yaw_count = 0
fig_yaw_count = 1


def main():

    df = pd.read_csv(path)
    d = df.groupby('trip', sort=False)
    df_list = [d.get_group(x) for x in d.groups]

    # trip info
    print([int(item['trip'][0:1]) for item in df_list])
    print('numbers of trips: ', len(df_list))
    ct = 0

    for df in df_list:
        ct += 1
        speed = np.array(df.speed)
        acc = np.array(df.Ax)
        yaw_rate = np.array(df.yaw_rate)
        speed_sec, acc_sec, yaw_sec = [], [], []
        speed_ltr, acc_ltr, yaw_ltr = [], [], []
        speed_int = []

        rows_left = len(df) % 10
        length = len(df)/10

        # append rows by second
        for index in range(1, length+1):
            spd_avg = np.mean(speed[10*(index-1):10*index])
            acc_avg = np.mean(acc[10*(index-1):10*index])
            yaw_avg = np.mean(yaw_rate[10*(index-1):10*index])

            speed_sec.append(spd_avg)
            acc_sec.append(acc_avg)
            yaw_sec.append(yaw_avg)

        # append rows left in the end
        if rows_left != 0:
            speed_sec.append(np.mean(speed[-rows_left:]))
            acc_sec.append(np.mean(acc[-rows_left:]))
            yaw_sec.append(np.mean(yaw_rate[-rows_left:]))

        # the speed_plot function runs very slow, but not sure the reason...
        speed_plot(df, speed_sec, speed_ltr, speed_int)
        acc_plot(df, acc_sec, acc_ltr)
        yaw_plot(df, yaw_sec, yaw_ltr)


# draw plot for speed
def speed_plot(df, speed_sec, speed_ltr, speed_int):
    global fig_spd_count
    global trip_spd_count

    # make speed list by letter
    for num in speed_sec:
        if num == 0:
            speed_ltr.append('a')
            speed_int.append(0)
        elif num > 0 and num <= 16:
            speed_ltr.append('b')
            speed_int.append(16)
        elif num > 16 and num <= 32:
            speed_ltr.append('c')
            speed_int.append(32)
        elif num > 32 and num <= 48:
            speed_ltr.append('d')
            speed_int.append(48)
        elif num > 48 and num <= 64:
            speed_ltr.append('e')
            speed_int.append(64)
        elif num > 64 and num <= 80:
            speed_ltr.append('f')
            speed_int.append(80)
        elif num > 80 and num <= 97:
            speed_ltr.append('g')
            speed_int.append(97)
        elif num > 97 and num <= 113:
            speed_ltr.append('h')
            speed_int.append(113)
        elif num > 113:
            speed_ltr.append('i')
            speed_int.append(130)

    if trip_spd_count >= 28*fig_spd_count:
        fig_spd_count = fig_spd_count+1
    trip_spd_count = trip_spd_count+1

    if trip_spd_count <= 28*fig_spd_count:
        plt.figure(fig_spd_count)
        plt.subplots_adjust(hspace=.3)
        plt.suptitle(path[-11:-4]+'_speed_bySec_P'+str(fig_spd_count),
                     fontsize=14)
        plt.subplot(4, 7, trip_spd_count-28*(fig_spd_count-1))

        plt.plot(range(len(speed_int)), speed_int)
        plt.tick_params(axis='x', labelsize=8)
        plt.yticks(speed_int, speed_ltr)
        plt.gca().set_ylim([-10, 140])
        plt.grid(True)
        plt.title('trip_'+str(int(df.trip[:1])), fontsize=10)
        plt.show()


# draw plot for acc
def acc_plot(df, acc_sec, acc_ltr):
    global fig_acc_count
    global trip_acc_count

    for num in acc_sec:
        if num <= -0.6:
            acc_ltr.append('a')
        elif num > -0.6 and num <= -0.4:
            acc_ltr.append('b')
        elif num > -0.4 and num <= -0.3:
            acc_ltr.append('c')
        elif num > -0.3 and num <= -0.05:
            acc_ltr.append('d')
        elif num > -0.05 and num <= 0:
            acc_ltr.append('e')
        elif num > 0 and num <= 0.05:
            acc_ltr.append('f')
        elif num > 0.05 and num <= 0.3:
            acc_ltr.append('g')
        elif num > 0.3 and num <= 0.4:
            acc_ltr.append('h')
        elif num > 0.4 and num <= 0.6:
            acc_ltr.append('i')
        elif num > 0.6:
            acc_ltr.append('j')

    if trip_acc_count >= 28*fig_acc_count:
        fig_acc_count = fig_acc_count+1
    trip_acc_count = trip_acc_count+1

    if trip_acc_count <= 28*fig_acc_count:
        plt.figure(fig_acc_count+3)
        plt.subplots_adjust(hspace=.3)
        plt.suptitle(path[-11:-4]+'_acc_bySec_P'+str(fig_acc_count),
                     fontsize=14)
        plt.subplot(4, 7, trip_acc_count-28*(fig_acc_count-1))

        ct_list = [acc_ltr.count('a'), acc_ltr.count('b'), acc_ltr.count('c'),
                   acc_ltr.count('d'), acc_ltr.count('e'), acc_ltr.count('f'),
                   acc_ltr.count('g'), acc_ltr.count('h'), acc_ltr.count('i'),
                   acc_ltr.count('j')]

        ltr_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        for index in range(len(ct_list)):
            if ct_list[index] == 0:
                ltr_list[index] = ' '

        plt.bar(range(10), ct_list, align='center', tick_label=True,
                capsize=True)
        plt.xticks(range(10), ltr_list)
        plt.tick_params(axis='x', labelsize=8)
        plt.tick_params(axis='y', labelsize=8)
        plt.grid(True)
        plt.title('trip_'+str(int(df.trip[:1])), fontsize=10)
        plt.show()


# draw plot for yaw rate
def yaw_plot(df, yaw_sec, yaw_ltr):
    global fig_yaw_count
    global trip_yaw_count

    for num in yaw_sec:
        if num < -0.6:
            yaw_ltr.append('a')
        elif num > -0.6 and num < -0.4:
            yaw_ltr.append('b')
        elif num > -0.4 and num < -0.3:
            yaw_ltr.append('c')
        elif num > -0.3 and num < -0.05:
            yaw_ltr.append('d')
        elif num > -0.05 and num < 0.05:
            yaw_ltr.append('e')
        elif num > 0.05 and num < 0.3:
            yaw_ltr.append('f')
        elif num > 0.3 and num < 0.4:
            yaw_ltr.append('g')
        elif num > 0.4 and num < 0.6:
            yaw_ltr.append('h')
        elif num > 0.6:
            yaw_ltr.append('i')

    if trip_yaw_count >= 28*fig_yaw_count:
        fig_yaw_count = fig_yaw_count+1
    trip_yaw_count = trip_yaw_count+1

    if trip_yaw_count <= 28*fig_yaw_count:
        plt.figure(fig_yaw_count+6)
        plt.subplots_adjust(hspace=.3)
        plt.suptitle(path[-11:-4]+'_yaw_bySec_P'+str(fig_yaw_count),
                     fontsize=14)
        plt.subplot(4, 7, trip_yaw_count-28*(fig_yaw_count-1))

        ct_list = [yaw_ltr.count('a'), yaw_ltr.count('b'), yaw_ltr.count('c'),
                   yaw_ltr.count('d'), yaw_ltr.count('e'), yaw_ltr.count('f'),
                   yaw_ltr.count('g'), yaw_ltr.count('h'), yaw_ltr.count('i')]

        ltr_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        for index in range(len(ct_list)):
            if ct_list[index] == 0:
                ltr_list[index] = ' '

        plt.bar(range(9), ct_list, align='center', tick_label=True,
                capsize=True)
        plt.xticks(range(9), ltr_list)
        plt.tick_params(axis='x', labelsize=8)
        plt.tick_params(axis='y', labelsize=8)
        plt.grid(True)
        plt.title('trip_'+str(int(df.trip[:1])), fontsize=10)
        plt.show()


main()
