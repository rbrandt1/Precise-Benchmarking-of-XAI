import pandas as pd
import numpy as np

folder = "Normalized-yes_Cuda-no"
num_runs = 3

for name in ['table_results_','table_times_']: 

    dfs = []

    for i in range(num_runs):
        dfs.append(pd.read_csv(r''+folder+'/Run '+str(i+1)+'/'+name+'.csv', index_col=False))
       
    #print(dfs)
    #exit()
       
    df = pd.concat(dfs)
    df.rename(columns={"Unnamed: 0": "Metric"}, inplace=True)

    by_row_index = df.groupby(df.Metric, sort=False)
    
    df_means = by_row_index.mean()
    
    #print(df_means)
    #exit()
    
    if name == 'table_times_':
        df_means = df_means.mean(axis=1)
     
    #print(df_means)
    #exit()
     
    df_std = by_row_index.std()

    #print(df_std)
    #exit()

    if name == 'table_times_':
        df_std = df_std.mean(axis=1)

    #print(df_std)
    #exit()

    df = df_means.round(2).astype(str) + u"\u00B1" + df_std.round(2).astype(str)
    
    #print(df)
    
    if name == 'table_times_':
        df = pd.DataFrame(df)
     
        df.insert(1, "Means", df_means, True)
        
        #print(df)
        #exit()
        
        df = df.sort_values(by=['Means'])
        
        #print(df)
        #exit()
        
        df =  df.drop(columns=['Means'])
        print(df.to_latex(index=True))
        
    else:
        
        #print(df)
    
        df['Delta'] = df_means.max(axis=1)-df_means.min(axis=1)
        df['Delta'] = df['Delta'].round(2).astype(str)
              
        #print(df)
        #exit()
              
        df = df.apply(lambda x: x.apply(lambda x: x.lstrip('0')))
        
        print(df.to_latex(index=True))
        
        
        #df = df.sort_values(by=['Delta'])
        
        #print(df.to_latex(index=True))