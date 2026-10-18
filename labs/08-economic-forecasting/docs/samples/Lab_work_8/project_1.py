#------------------- МІНІ-проект комплексний аналіз динаміки продажів торгівельної компанії Частина_1 -----------------------

'''

Завдання:
1. Побудувати динаміку зміни прибутку за рік та здійснити прогноз прибутку на 6 місяців наступного року
2. Визначити динаміку тренду за кількістю продажів товару N протягом року та прогнозні показники на 6 місяців наступного року
3. Побудувати динаміку  продажів (оборудку) по місяцях (табличка, діаграма)
4. Побудувати динаміку прибутку  по місяцях (табличка, діаграма)
5. Побудувати динаміку продажів та прибутку (табличка, діаграма) за Регіонами

БАЗОВИЙ ПІДХІД - методи статистичного навчання (прикладний статистичний аналіз)

'''


#--------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as mt
import warnings
warnings.filterwarnings('ignore')

#-------------------- ЕТАП І ЛІНІЙНИЙ ЕКОНОМІЧНИЙ АНАЛІЗ п.п. 3,4 ------------------------------
#----------------- розділення вхідного масиву на кластери за місяцями --------------------------
def segment_month(d, index):
    global F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12
    n = d[index].size                       # розмір стовпця 'Цена реализации'
    print('n=', n)
    F0=np.zeros((n))                        # масиви даних за місяцями
    # сегмент продажів за січень -------------------------------------------------
    i=0; j=0; l=0
    while d['Місяц'][i]=='Січень':
        F0[j]=d[index][i]
        i=i+1; j=j+1; l=l+1
    F1 = np.zeros((l))
    for l1 in range (0,l):
        F1[l1]=F0[l1]
    print('F1=',F1)
    # сегмент продажів за лютий -----------------------------------------------
    j=0; l=0
    while d['Місяц'][i]=='Лютий':
        F0[j]=d[index][i]
        i=i+1; j=j+1; l=l+1
    F2 = np.zeros((l))
    for l1 in range (0,l):
        F2[l1]=F0[l1]
    print('F2=',F2)
    # сегмент продажів за березень ---------------------------------------------------
    j = 0; l = 0
    while d['Місяц'][i] == 'Березень':
        F0[j] = d[index][i]
        i = i + 1; j = j + 1; l = l + 1
    F3 = np.zeros((l))
    for l1 in range(0, l):
        F3[l1] = F0[l1]
    print('F3=', F3)
    # сегмент продажів за квітень ------------------------------------------------
    j = 0; l = 0
    while d['Місяц'][i] == 'Квітень':
        F0[j] = d[index][i]
        i = i + 1; j = j + 1; l = l + 1
    F4 = np.zeros((l))
    for l1 in range(0, l):
        F4[l1] = F0[l1]
    print('F4=', F4)
    # сегмент продажів за травень ---------------------------------------------------
    j = 0; l = 0
    while d['Місяц'][i] == 'Травень':
        F0[j] = d[index][i]
        i = i + 1; j = j + 1; l = l + 1
    F5 = np.zeros((l))
    for l1 in range(0, l):
        F5[l1] = F0[l1]
    print('F5=', F5)
    # сегмент продажів за червень --------------------------------------------------
    j = 0; l = 0
    while d['Місяц'][i] == 'Червень':
        F0[j] = d[index][i]
        i = i + 1; j = j + 1; l = l + 1
    F6 = np.zeros((l))
    for l1 in range(0, l):
        F6[l1] = F0[l1]
    print('F6=', F6)
    # сегмент продажів за липень --------------------------------------------------
    j = 0; l = 0
    while d['Місяц'][i] == 'Липень':
        F0[j] = d[index][i]
        i = i + 1; j = j + 1; l = l + 1
    F7 = np.zeros((l))
    for l1 in range(0, l):
        F7[l1] = F0[l1]
    print('F7=', F7)
    # сегмент продажів за серпень --------------------------------------------------
    j = 0; l = 0
    while d['Місяц'][i] == 'Серпень':
        F0[j] = d[index][i]
        i = i + 1; j = j + 1; l = l + 1
    F8 = np.zeros((l))
    for l1 in range(0, l):
        F8[l1] = F0[l1]
    print('F8=', F8)
    # сегмент продажів за вересень --------------------------------------------------
    j = 0; l = 0
    while d['Місяц'][i] == 'Вересень':
        F0[j] = d[index][i]
        i = i + 1; j = j + 1; l = l + 1
    F9 = np.zeros((l))
    for l1 in range(0, l):
        F9[l1] = F0[l1]
    print('F9=', F9)
    # сегмент продажів за жовтень --------------------------------------------------
    j = 0; l = 0
    while d['Місяц'][i] == 'Жовтень':
        F0[j] = d[index][i]
        i = i + 1; j = j + 1; l = l + 1
    F10 = np.zeros((l))
    for l1 in range(0, l):
        F10[l1] = F0[l1]
    print('F10=', F10)
    # сегмент продажів за грудень --------------------------------------------------
    j = 0; l = 0
    while d['Місяц'][i] == 'Грудень':
        F0[j] = d[index][i]
        i = i + 1; j = j + 1; l = l + 1
    F12 = np.zeros((l))
    for l1 in range(0, l):
        F12[l1] = F0[l1]
    print('F12=', F12)
    # print('i=', i)
    # сегмент продажів за листопад --------------------------------------------------
    j = 0; l = 0
    while d['Місяц'][i] == 'Листопад':
        F0[j] = d[index][i]
        i = i + 1; j = j + 1; l = l + 1
    F11 = np.zeros((l))
    for l1 in range(0, l):
        F11[l1] = F0[l1]
    print('F11=', F11)
    return F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12

#----------------- розрахунок сумарного значення економічного показника за місяць --------------------------
def Sum_segment_month(d, index, F_in):
    global F_segment_month
    n = d[index].size                                 # розмір стовпця 'index'
    F0=np.zeros((n)); F_segment_month=np.zeros((12))  # масиви даних за місяцями
    # сегмент продажів за січень -------------------------------------------------
    i=0; j=0; l=0
    while d['Місяц'][i]=='Січень':
        F0[j] = F_in[i]
        i=i+1; j=j+1; l=l+1
    for l1 in range (0,l):
        F_segment_month[0]=F_segment_month[0]+F0[l1]
    # сегмент продажів за лютий -----------------------------------------------
    j=0; l=0
    while d['Місяц'][i]=='Лютий':
        F0[j] = F_in[i]
        i=i+1; j=j+1; l=l+1
    for l1 in range (0,l):
        F_segment_month[1] = F_segment_month[1] + F0[l1]
    # сегмент продажів за березень ---------------------------------------------------
    j = 0; l = 0
    while d['Місяц'][i] == 'Березень':
        F0[j] = F_in[i]
        i = i + 1; j = j + 1; l = l + 1
    for l1 in range(0, l):
        F_segment_month[2] = F_segment_month[2] + F0[l1]
    # сегмент продажів за квітень ------------------------------------------------
    j = 0; l = 0
    while d['Місяц'][i] == 'Квітень':
        F0[j] = F_in[i]
        i = i + 1; j = j + 1; l = l + 1
    for l1 in range(0, l):
        F_segment_month[3] = F_segment_month[3] + F0[l1]
    # сегмент продажів за травень ---------------------------------------------------
    j = 0; l = 0
    while d['Місяц'][i] == 'Травень':
        F0[j] = F_in[i]
        i = i + 1; j = j + 1; l = l + 1
    for l1 in range(0, l):
        F_segment_month[4] = F_segment_month[4] + F0[l1]
    # сегмент продажів за червень --------------------------------------------------
    j = 0; l = 0
    while d['Місяц'][i] == 'Червень':
        F0[j] = F_in[i]
        i = i + 1; j = j + 1; l = l + 1
    for l1 in range(0, l):
        F_segment_month[5] = F_segment_month[5] + F0[l1]
    # сегмент продажів за липень --------------------------------------------------
    j = 0; l = 0
    while d['Місяц'][i] == 'Липень':
        F0[j] = F_in[i]
        i = i + 1; j = j + 1; l = l + 1
    for l1 in range(0, l):
        F_segment_month[6] = F_segment_month[6] + F0[l1]
    # сегмент продажів за серпень --------------------------------------------------
    j = 0; l = 0
    while d['Місяц'][i] == 'Серпень':
        F0[j] = F_in[i]
        i = i + 1; j = j + 1; l = l + 1
    for l1 in range(0, l):
        F_segment_month[7] = F_segment_month[7] + F0[l1]
    # сегмент продажів за вересень --------------------------------------------------
    j = 0; l = 0
    while d['Місяц'][i] == 'Вересень':
        F0[j] = F_in[i]
        i = i + 1; j = j + 1; l = l + 1
    for l1 in range(0, l):
        F_segment_month[8] = F_segment_month[8] + F0[l1]
    # сегмент продажів за жовтень --------------------------------------------------
    j = 0; l = 0
    while d['Місяц'][i] == 'Жовтень':
        F0[j] = F_in[i]
        i = i + 1; j = j + 1; l = l + 1
    for l1 in range(0, l):
        F_segment_month[9] = F_segment_month[9] + F0[l1]
    # сегмент продажів за грудень --------------------------------------------------
    j = 0; l = 0
    while d['Місяц'][i] == 'Грудень':
        F0[j] = F_in[i]
        i = i + 1; j = j + 1; l = l + 1
    for l1 in range(0, l):
        F_segment_month[11] = F_segment_month[11] + F0[l1]
    # сегмент продажів за листопад --------------------------------------------------
    j = 0; l = 0
    while d['Місяц'][i] == 'Листопад':
        F0[j] = F_in[i]
        i = i + 1; j = j + 1; l = l + 1
    for l1 in range(0, l):
        F_segment_month[10] = F_segment_month[10] + F0[l1]
    print('F_segment_month=', F_segment_month)
    return  F_segment_month

#------------------------- розрахунок продажів (оборутку) -----------------------------
def sale(d):
    global F_sale
    n = d['КільКість реалізацій'].size  # розмір стовпця
    F_sale = np.zeros((n))
    for i in range(0, n):
        F_sale[i] = d['КільКість реалізацій'][i]*d['Собівартість одиниці'][i]
    print('F_sale=', F_sale)
    return F_sale

#------------------------------ розрахунок прибутку -----------------------------------
def profit(d):
    global F_profit
    n = d['КільКість реалізацій'].size  # розмір стовпця
    F_profit = np.zeros((n))
    for i in range(0, n):
        F_profit[i] = d['КільКість реалізацій'][i]*(d['Ціна реализації'][i]-d['Собівартість одиниці'][i])
    print('F_profit=', F_profit)
    return F_profit

#------------------------------------парсинг вхідного файла -------------------------------

# пряме зчитування
dateparse = lambda x: pd.to_datetime(x, dayfirst=True)
d = pd.read_excel('Pr_1.xls', parse_dates=['Дата'], date_parser=dateparse)
# парсінг та індексація за датою
dd = pd.read_excel('Pr_1.xls', parse_dates=['Дата'], index_col='Дата', date_parser=dateparse)

print('d=', d)                                          # вивід усього масиву файлу Pr12.xls
print('dd=', dd)
print('---------------- Ціна реализації -------------------')
print(d['Ціна реализації'])                             # вивід стовпця 'Цена реализации'
print(type(float(d['Ціна реализації'][0])))             # тип елементу стовпця 'Цена реализации'

# ----------------------------- основні виклики  --------------------------------------
# 0. візуальний аналіз вхідних даних  -----------------------------
index='Ціна реализації'
segment_month(d, index)                             # Сегментація вхідного масиву на місяця
plt.title(index); d[index].plot()                   # стовпчик index з аргументом 0-n
plt.show()
plt.title(index); dd[index].plot()                  # стовпчик index з аргументом ДАТА
plt.show()
plt.title('Month_1-6')
plt.plot(F1); plt.plot(F2); plt.plot(F3)            # вибірки за місяцями
plt.plot(F4); plt.plot(F5); plt.plot(F6)
plt.show()
# 1. Розрахунок продажів --------------------------------------------
sale(d)                                    # за рік
Sum_segment_month(d, index, F_sale)        # по місяцях
plt.title('Sale'); plt.plot(F_sale)        # візуалізація за рік
plt.show()
s=pd.Series(F_segment_month)
plt.title('Sale'); s.plot(kind='bar')      # візуалізація по місяцях
plt.show()
F_segment_month_sale=F_segment_month
# 2. Розрахунок прибутку --------------------------------------------
profit(d)                                 # за рік
Sum_segment_month(d, index, F_profit)     # по місяцях
plt.title('Profit'); plt.plot(F_profit)   # візуалізація за рік
plt.show()
s=pd.Series(F_segment_month)
plt.title('Profit'); s.plot(kind='bar')   # візуалізація по місяцях
plt.show()
F_segment_month_profit=F_segment_month
# 3. Інтеграція продаж + прибуток -----------------------------------
plt.title('Sale + Profit')
plt.plot(F_sale)
plt.plot(F_profit)
plt.show()
s1=pd.Series(F_segment_month_profit)
s2=pd.Series(F_segment_month_sale)
plt.title('Sale + Profit')
s2.plot(kind='bar', color='b')
s1.plot(kind='bar', color='g')

plt.show()
#-------------------- ЕТАП ІІ СТАТИСТИЧНИЙ ЕКОНОМІЧНИЙ АНАЛІЗ п.п. 1,2 --------------------------
# вихідні дані:
# F_sale  -  продажі за рік; F_segment_month_sale  -  продажі по місяцях
# F_profit - продажі за рік; F_segment_month_profit - продажі по місяцях
# ------------------- розрахунок статистичних характеристик вибірки -----------------------------

def Stat_characteristics (SL, Text):
    # статистичні характеристики вибірки з урахуванням тренду
    Yout = MNK_Stat_characteristics(SL)
    iter = len(Yout)
    SL0 = np.zeros((iter ))
    for i in range(iter):
        SL0[i] = SL[i] - Yout[i, 0]
    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)
    print('------------', Text ,'-------------')
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('-----------------------------------------------------')
    plt.title(Text)
    plt.hist(SL,  bins=30, facecolor="blue", alpha=0.5) # гістограма закону розподілу ВВ
    plt.show()
    return

# ------------- МНК згладжуваннядля визначення стат. характеристик -------------
def MNK_Stat_characteristics (S0):
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT=F.T
    FFT = FT.dot(F)
    FFTI=np.linalg.inv(FFT)
    FFTIFT=FFTI.dot(FT)
    C=FFTIFT.dot(Yin)
    Yout=F.dot(C)
    return Yout

# ------------------------------ МНК згладжування -------------------------------------
def MNK (Y_coord):
    # ---- формування структури вхідних матриць МНК ------
    iter = Y_coord.size
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 5))
    for i in range(iter):
        Yin[i, 0] = Y_coord[i]
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
        F[i, 3] = float(i * i * i)
        F[i, 4] = float(i * i * i * i)
    # ------------- алгоритм МНК ------------------------
    FT=F.T
    FFT = FT.dot(F)
    FFTI=np.linalg.inv(FFT)
    FFTIFT=FFTI.dot(FT)
    C=FFTIFT.dot(Yin)
    Yout=F.dot(C)
    return Yout

# ------------------------------ МНК прогноз -------------------------------------
def MNK_extrapolation (Y_coord, koef):
    # ---- формування структури вхідних матриць МНК ------
    iter = Y_coord.size
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 5))
    for i in range(iter):
        Yin[i, 0] = Y_coord[i]
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
        F[i, 3] = float(i * i * i)
        F[i, 4] = float(i * i * i * i)
    # ------------- алгоритм МНК ------------------------
    FT=F.T
    FFT = FT.dot(F)
    FFTI=np.linalg.inv(FFT)
    FFTIFT=FFTI.dot(FT)
    C=FFTIFT.dot(Yin)
    Yout=F.dot(C)
    j=koef
    for i in range(0, koef):
        Yout[i, 0] = C[0, 0] + C[1, 0]*j + (C[2, 0]*j*j) + (C[3, 0]*j*j*j) + (C[4, 0]*j*j*j*j)
        j = j+1

    return Yout

# ---------------------------- ОЦІНЮВАННЯ ------------------------------
# ---------------------- МНК для продажів за рік -----------------------
Yout0 = MNK (F_sale)
print('------------ вхідна вибірка  ----------')
Stat_characteristics (F_sale, 'вхідна вибірка за рік')
print('-------------- МНК оцінка  ------------')
Stat_characteristics (Yout0, 'МНК оцінка за рік')
plt.title('MNK_sale')
plt.plot(F_sale)
plt.plot(Yout0)
plt.show()

# ---------------------- МНК для продажів за місяцями -------------------
Yout1 = MNK (F_segment_month_sale)
print('------------ вхідна вибірка  ----------')
Stat_characteristics (F_segment_month_sale, 'вхідна вибірка за місяцями')
print('-------------- МНК оцінка  ------------')
Stat_characteristics (Yout1, 'МНК оцінка за місяцями')
plt.title('MNK_segment_month_sale')
plt.plot(F_segment_month_sale)
plt.plot(Yout1)
plt.show()

# ----------------------------- ПРОГНОЗ --------------------------------
# ---------------------- МНК для продажів за рік -----------------------
prognoz=int (0.5*(F_sale.size))
Graf_Yout0=np.zeros(prognoz)
Graf_Yout0_1=np.zeros(6)
Yout0 = MNK_extrapolation (F_sale, prognoz)
Graf_Yout0_1[0]=Yout0[1, 0]
Graf_Yout0_1[1]=Yout0[(int(prognoz/6)), 0]
Graf_Yout0_1[2]=Yout0[(int((2*prognoz/6))), 0]
Graf_Yout0_1[3]= Yout0[(int((3*prognoz/6))), 0]
Graf_Yout0_1[4]=Yout0[(int((4*prognoz/6))), 0]
Graf_Yout0_1[5]=Yout0[(int((5*prognoz/6))), 0]

print('------------ MNK_extrapolation_sale ---------------')
for i in range(0 , 6):
    print ('Yout0[', i, ',0]=', Graf_Yout0_1[i])
plt.title('MNK_extrapolation_sale')
for i in range(0, prognoz):
    Graf_Yout0[i] = Yout0[i, 0]
plt.plot(Graf_Yout0)
plt.show()

# ---------------------- МНК для продажів за місяцями -------------------
prognoz=6
Graf_Yout1=np.zeros(prognoz)
Yout1 = MNK_extrapolation  (F_segment_month_sale, prognoz)
print('------ MNK_extrapolation_segment_month_sale --------')
for i in range(prognoz):
    print ('Yout1[', i, ',0]=', Yout1[i,0])
plt.title('MNK_extrapolation_segment_month_sale')
for i in range(0, prognoz):
    Graf_Yout1[i] = Yout1[i, 0]
plt.plot(Graf_Yout1)
plt.show()

# -------------------------------------------------------------------------
plt.title('MNK_extrapolation за рік та місяці')
plt.plot(Graf_Yout0_1)
plt.plot(Graf_Yout1)
plt.show()