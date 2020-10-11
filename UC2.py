#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:13:59 2019

@author: tzm
"""
#%%
import pulp
import numpy as np

# =============================================================================
# J 机组数 100
# K 时间段数 24
# NL 耗量近似段数 8
# 选装备用 10%
# 生产费用近似段 4
# 开机费用近似段 15
# =============================================================================

#%% define function
def PC(a_input,b_input,c_input,p_input):
    result=a_input+b_input*p_input+c_input*p_input*p_input
    return result;

#%% constants
NL=4#出力近似分段为4段
ND=15#开机费用近似为15段

#system_data_1=[\P最大出力,P\最小出力,UT最小开机时间,DT最小关机时间,inistate初始运行状况]
system_data_1_0=np.array([[455,150,8,8,8],
                       [455,150,8,8,8],
                       [130,20,5,5,-5],
                       [130,20,5,5,-5],
                       [162,25,6,6,-6],
                       [80,20,3,3,-3],
                       [85,25,3,3,-3],
                       [55,10,1,1,-1],
                       [55,10,1,1,-1],
                       [55,10,1,1,-1]])
system_data_1=np.tile(system_data_1_0,(10,1))


RU= np.rint((system_data_1[:,0]-system_data_1[:,1])*0.6) #上升斜率约束
RU=RU.astype(np.int)


SU=np.rint( (system_data_1[:,0]-system_data_1[:,1])*0.7 )#开机上升斜率约束
SU=SU.astype(np.int)
RD=np.rint( (system_data_1[:,0]-system_data_1[:,1])*0.6 )#下降斜率约束
RD=RD.astype(np.int)
SD=np.rint( (system_data_1[:,0]-system_data_1[:,1])*0.8 )#关机下降斜率约束
SD=SD.astype(np.int)
    
    
    

#system_data_2=[a,b,c,hc,cc,t_cold]
system_data_2_0=np.array([[1000,16.19,0.00048,4500,9000,5],
                        [970,17.26,0.00031,5000,10000,5],
                        [700,16.60,0.00200,550,1100,4],
                        [680,16.50,0.00211,560,1120,4],
                        [450,19.70,0.00398,900,1800,4],
                        [370,22.26,0.00712,170,340,2],
                        [480,27.74,0.00079,260,520,2],
                        [660,25.92,0.00413,30,60,0],
                        [665,27.27,0.00222,30,60,0],
                        [670,27.79,0.00173,30,60,0]])
system_data_2=np.tile(system_data_2_0,(10,1))
#load_demand
load_demand_0=np.array([700,750,850,950,1000,1100,1150,1200,1300,1400,1450,1500,
                     1400,1300,1200,1050,1000,1100,1200,1400,1300,1100,900
                     ,800])[np.newaxis,:]
load_demand=np.tile(load_demand_0)
print(load_demand)

'''

#%% formed constrants / constrants matrix
# =============================================================================
# 形成系数A矩阵
# =============================================================================
productcost_A=np.zeros((system_data_1.shape[0],1))
#print(A.shape[0])
for i in range(0,productcost_A.shape[0]):
#    print(system_data_2[i,0]) a
#    print(system_data_2[i,1]) b
#    print(system_data_2[i,2]) c
#    print(system_data_1[i,1]) P\
    productcost_A[i,0]=system_data_2[i,0]+system_data_2[i,1]*system_data_1[i,1]+system_data_2[i,2]*system_data_1[i,1]**2

# =============================================================================
# 形成K_j_t 矩阵   dimension：10*15
# =============================================================================
#分段时间矩阵 t_cold + DT
#print(system_data_2[:,5]) #t_cold
#print(system_data_2[:,4]) cc
#print(system_data_2[:,3]) hc
#print(system_data_1[:,3]) #DT
t_interval=system_data_2[:,5]+system_data_1[:,3]
#print(t_interval)
#np.zeros((system_data_1.shape[0],1))
starup_K=np.zeros((system_data_1.shape[0],15))
for i in range(0,system_data_1.shape[0]):
    for j in range(0,15):
        if  (j+1)>=1 and (j+1)<=t_interval[i]:
            starup_K[i,j]=system_data_2[i,3]
        else:
            starup_K[i,j]=system_data_2[i,4]
#print(starup_K)

# =============================================================================
# 关机费用       
# =============================================================================
shutdown_cd=np.zeros((system_data_1.shape[0],1))      

# =============================================================================
# 生产费用分段线性化处理
# =============================================================================
#出力分段 10*5
power_interval_T=np.zeros((system_data_1.shape[0],NL+1))
#print(system_data_1[0:10,0]) #\P
#(system_data_1[0:10,1]) #P\
for i in range(0,system_data_1.shape[0]):
    power_interval_T[i,:]=np.linspace(system_data_1[i,1],system_data_1[i,0],NL+1,dtype=np.float64)
#print(power_interval_T)

#分段斜率 slope_F  10*4/NL
slop_F=np.zeros((len(system_data_1),NL))
for i in range(len(system_data_1)):
    for j in range(NL):
        power_large=PC(system_data_2[i,0],system_data_2[i,1],system_data_2[i,2],power_interval_T[i,j+1])
        power_low=PC(system_data_2[i,0],system_data_2[i,1],system_data_2[i,2],power_interval_T[i,j])
        power_gap=power_interval_T[i,j+1]-power_interval_T[i,j]
        slop_F[i,j]=(power_large-power_low)/power_gap


# =============================================================================
# 初始开机时间 初始关机时间
# =============================================================================
# =============================================================================
# system_data_init=system_data_1[:,4]#中间变量
# init_open=np.copy(system_data_init)
# init_close=np.copy(system_data_init)
# 
# for i in range(len(system_data_init)):
#     if system_data_init[i]>0:
#         init_close[i]=0
#     elif system_data_init[i]<=0:
#         init_open[i]=0
# =============================================================================


#%% variables

# =============================================================================
# binary variables
# =============================================================================
onoffstage_v=np.zeros((system_data_1.shape[0],24))
row_matrix_v = len(onoffstage_v) 
col_matrix_v = len(onoffstage_v[0])
var_v = pulp.LpVariable.dicts("v", (range(row_matrix_v), range(col_matrix_v)),0,1,pulp.LpBinary) 
#print(var_mu[0][0].varValue) #查看取值？

# =============================================================================
# 生产函数变量 c_p 10*24 
# =============================================================================
productcost_cp=np.zeros((system_data_1.shape[0],24))
row_matrix_cp = len(productcost_cp) 
col_matrix_cp = len(productcost_cp[0])
var_cp = pulp.LpVariable.dicts("cp", (range(row_matrix_cp), range(col_matrix_cp)), lowBound = 0) 

# =============================================================================
# 生产函数变量 c_u 10*24 
# =============================================================================
starupcost_cu=np.zeros((system_data_1.shape[0],24))
row_matrix_cu = len(starupcost_cu) 
col_matrix_cu = len(starupcost_cu[0])
var_cu = pulp.LpVariable.dicts("cu", (range(row_matrix_cu), range(col_matrix_cu)), lowBound = 0) 

# =============================================================================
# 出力变量 p 10*24
# =============================================================================
powerprovide_p=np.zeros((system_data_1.shape[0],24))
row_matrix_p = len(powerprovide_p) 
col_matrix_p = len(powerprovide_p[0])
var_p = pulp.LpVariable.dicts("p", (range(row_matrix_p), range(col_matrix_p)), lowBound = 0) 

# =============================================================================
# 最大出力变量 p_max 10*24
# =============================================================================
powermax_p_max=np.zeros((system_data_1.shape[0],24))
row_matrix_p_max = len(powermax_p_max) 
col_matrix_p_max = len(powermax_p_max[0])
var_p_max = pulp.LpVariable.dicts("p_max", (range(row_matrix_p_max), range(col_matrix_p_max)), lowBound = 0) 

# =============================================================================
# 分段出力能力变量 q_1 10*24*4
# =============================================================================
power_q = pulp.LpVariable.dicts("q", (range(len(system_data_1)), range(24),range(NL)), lowBound = 0) 


#%% 设置初始开关机状态

for i in range(len(system_data_1[:,4])):
    if system_data_1[i,4]>0:
        var_v[i][0]=1
    elif system_data_1[i,4]<=0:
        var_v[i][0]=0


#%% 定义要优化的问题
UC = pulp.LpProblem('UC_task', sense=pulp.LpMinimize) 

#%% 目标函数
UC += (pulp.lpSum( (pulp.lpSum(  (var_cp[j][k]+var_cu[j][k] ) for j in range(row_matrix_cu)   ))  for k in range(col_matrix_cu)))
#(var_cp[j][k] + var_cu[j][k])
#%% 
# =============================================================================
# 出力约束
# =============================================================================
for k in range(col_matrix_p):
    UC +=(pulp.lpSum(var_p[j][k] for j in range(row_matrix_p))==load_demand[0,k])


# =============================================================================
# 最大出力约束
# =============================================================================
for j in range(col_matrix_p_max):
    UC +=(pulp.lpSum(var_p_max[i][j] for i in range(row_matrix_p_max))>=1.1*load_demand[0,j])





#%% 目标函数约束
# =============================================================================
# 出力分段约束
# =============================================================================

#式子6
for j in range(row_matrix_cp):
    for k in range(col_matrix_cp):
        UC += (   var_cp[j][k] == (  productcost_A[j,0]*var_v[j][k] + ( pulp.lpSum(slop_F[j,l]*power_q[j][k][l] for l in range(NL)) ))   )

#式子7
for j in range(row_matrix_cp):
    for k in range(col_matrix_cp):
        UC += ((system_data_1[j,1]*var_v[j][k]+(pulp.lpSum(power_q[j][k][l] for l in range(NL))))==var_p[j][k])

#式子8
for j in range(row_matrix_cp):
    for k in range(col_matrix_cp):
        UC += (power_q[j][k][0] <= power_interval_T[j,0]-system_data_1[j,1])

#式子9
for j in range(row_matrix_cp):
    for k in range(24):
        for l in range(2,NL-1):
            UC += power_q[j][k][l] <= power_interval_T[j,l] -power_interval_T[j,l-1]
#式子10
for j in range(row_matrix_cp):
    for k in range(col_matrix_cp):
        UC +=power_q[j][k][NL-1]<=system_data_1[j,0]-power_interval_T[j,NL-1]


# =============================================================================
# 开机分段约束
# =============================================================================
for j in range(row_matrix_cu):
    for k in range(col_matrix_cu):
        for t in range(min(ND,k)):
            UC +=var_cu[j][k]>=(starup_K[j,t]*( var_v[j][k] - pulp.lpSum( var_v[j][k-n] for n in range(1,t)) ))

#%% 火力机组约束

# =============================================================================
# 启停机约束  
# =============================================================================
for j in range(row_matrix_cp):
    for k in range(col_matrix_cp):
        UC +=(var_p[j][k] <= var_p_max[j][k])
        UC +=(var_p[j][k] >= var_v[j][k]*system_data_1[j,1])
        UC +=(var_p_max[j][k]<=system_data_1[j,0]*var_v[j][k])

# =============================================================================
# 斜率约束
# =============================================================================
#式子18
for j in range(row_matrix_cp):
    for k in range(1,col_matrix_cp):
        UC +=var_p_max[j][k] <= var_p[j][k-1]+RU[j]*var_v[j][k-1]+SU[j]*[var_v[j][k]-var_v[j][k-1]]+system_data_1[j,0]*(1-var_v[j][k])
#式子19
for j in range(row_matrix_cp):
    for k in range(0,col_matrix_cp-1):
        UC +=var_p_max[j][k] <= system_data_1[j,0]*var_v[j][k+1]+SD[j]*(var_v[j][k]-var_v[j][k+1])
#式子20
for j in range(row_matrix_cp):
    for k in range(1,col_matrix_cp):
        UC +=var_p[j][k-1]-var_p[j][k] <= RD[j]*var_v[j][k]+SD[j]*[var_v[j][k-1]-var_v[j][k]]+system_data_1[j,0]*(1-var_v[j][k-1])


#%% 启停时间约束

# =============================================================================
# Gj表达式
# =============================================================================

initialvalue_G=np.copy(system_data_1[:,2]-system_data_1[:,4])
for i in range(10):
    initialvalue_G[i]=initialvalue_G[i]*var_v[i][0]

  
# =============================================================================
# 式21
# =============================================================================
for j in range(row_matrix_cp):
    UC +=(   pulp.lpSum(  (1-var_v[j][k]) for k in range(initialvalue_G[j])  )==0   )

# =============================================================================
# 式22 
# =============================================================================

for j in range(row_matrix_cp):
    for k in range((initialvalue_G[j]+1),(24+1-system_data_1[j,2])):
        UC +=(   pulp.lpSum( var_v[j][n] for n in range(initialvalue_G[j],(k+system_data_1[j,2]-1+1))  )>= (system_data_1[j,2]*(var_v[j][k]-var_v[j][k-1])) )     


# =============================================================================
# 式23
# =============================================================================
  
for j in range(10):
    for k in range((24-system_data_1[j,2]+1),(24)):
        UC +=(   pulp.lpSum(   (var_v[j][n]-(var_v[j][k]-var_v[j][k-1])) for n in range(k,24)  )>=0   )



# =============================================================================
# Lj 表达式
# =============================================================================

initialvalue_L=np.copy(system_data_1[:,3]+system_data_1[:,4])
for i in range(10):
    initialvalue_L[i]=initialvalue_L[i]*(1-var_v[i][0])


# =============================================================================
# 式子24
# =============================================================================

for j in range(row_matrix_cp):
    UC +=(   pulp.lpSum(  var_v[j][k] for k in range(initialvalue_L[j])  )==0   )


# =============================================================================
# 式子25
# =============================================================================

for j in range(row_matrix_cp):
    for k in range((initialvalue_L[j]+1),(24-system_data_1[j,3]+1)):        
        UC += (   pulp.lpSum(   (1-var_v[j][n]) for n in range(k,k+system_data_1[j,3]-1+1)   )>=system_data_1[j,3]*(var_v[j][k-1]-var_v[j][k])   )

# =============================================================================
# 式子26
# =============================================================================

for j in range(row_matrix_cp):
    for k in range((24-system_data_1[j,3]+2),(24)):
        UC += pulp.lpSum(   1-var_v[j][n]-(var_v[j][k-1]-var_v[j][k]) for n in range(k,24)  )>=0

     



#%% 求解

#模型求解
UC.solve(pulp.CPLEX())

print("Status:", pulp.LpStatus[UC.status])
#结果显示
# check status : pulp.LpStatus[PB.status]
print("\n","status:",pulp.LpStatus[UC.status],"\n")

for v in UC.variables():
    print("\t",v.name,"=",v.varValue,"\n")

print("Minimun Cost =",pulp.value(UC.objective))


'''


