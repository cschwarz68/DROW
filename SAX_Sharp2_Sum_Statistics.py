import pandas as pd
import numpy as np
import glob


filenames = glob.glob("H:\SHRP2\SAX_Sharp2\*.csv")
df_list = []

for filename in filenames:
    
    f = pd.read_csv(filename)
    print filename
    trip_id = filename[31:-4]
    veh_id = int(f.veh_id[:1])
    
    e_amt = f.speed_ltr.tolist().count('e')
    f_amt = f.speed_ltr.tolist().count('f')    
    g_amt = f.speed_ltr.tolist().count('g')
    h_amt = f.speed_ltr.tolist().count('h')
    i_amt = f.speed_ltr.tolist().count('i')
    totLen = float(len(f))
    spd_null_sum = f.speed_network.isnull().sum()
    acc_x = np.array(f.acc_x_ltr)
    scd = ''
    
    for index in range(len(acc_x)-1):
        if (acc_x[index] == 'h' or acc_x[index] == 'i' or acc_x[index] == 'j'):
            if scd == '':
                scd = scd +str(index+1)+'/'
            elif scd.count('/') ==1:
                pre_num = int(scd[:-1])
                scd = scd + str(index+1-pre_num)+'/' 
                pre_scd = index+1     
            elif scd.count('/') >1:               
                scd = scd + str(index+1 - pre_scd)+'/'
                pre_scd = index+1
    
    data = np.array([[trip_id, veh_id, (e_amt+f_amt+g_amt+h_amt+i_amt)/totLen, 
                (f_amt+g_amt+h_amt+i_amt)/totLen, (g_amt+h_amt+i_amt)/totLen, 
                (h_amt+i_amt)/totLen, i_amt/totLen, len(f), spd_null_sum, 
                spd_null_sum/totLen, scd]])    
    columns = ['trip_id', 'veh_id', 'e%_n_higher','f%_n_higher','g%_n_higher','h%_n_higher',
                'i%', 'total_rows', 'empty_spd', 'spd_miss%', 'high_acc']
    df = pd.DataFrame(data, columns=columns)  
    df_list.append(df)  
      
frame = pd.concat(df_list, axis = 0)
    
    
frame.to_csv("H:\SHRP2\SAX_shrp2_sum_stats.csv", index=None)    
    
    
    
    