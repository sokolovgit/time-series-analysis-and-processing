#---------------- МІНІ-проект комплексний аналіз динаміки продажів торгівельної компанії Частина_2 -------------

'''

Завдання:
1. Побудувати динаміку продажів на 13, 14 тижні
2. Оцінити вплив запуску промо акцій

БАЗОВИЙ ПІДХІД - методи статистичного навчання (прикладний статистичний аналіз)

'''

#--------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as mt
import warnings
warnings.filterwarnings('ignore')

#------------------------------------парсинг вхідного файла -------------------------------
# пряме зчитування вмісту файла Pr13.xls до DataFrame - d
d = pd.read_excel('Pr_2.xls', parse_dates=['Day'])
print('d=', d)                                           # вивід усього DataFrame файлу Pr12.xls
Title_d=d.columns                                        # імпорт імен стовпців DataFrame - d
print(Title_d)
print(Title_d[3])
index_d=Title_d[3]
print('----------------', index_d , '-------------------')
print(d[index_d])                                          # вивід стовпця index
print(type(float(d[index_d][0])))                          # тип елементу стовпця index


#-------------------- ЕТАП І очищення, візуалізація та сегментація даних ---------------------------
# 0.1 очищення та візуальний аналіз вхідних даних  ------------------------------
print('--------- d_segment[', index_d ,'] --------------')
d_segment=d[d.Sales != '?']                                # виключення строк із значенням ?
print('d_segment [', index_d , ']=', d_segment[index_d])
plt.title(index_d); d_segment[index_d].plot()              # візуалізація очищених даних
plt.show()

# 0.2 сегментація даних за тижнями  --------------------------------------------
d_segment_Week_2=d[d.Week == 2]                            # відбір строк за ознакою належності до тижня
d_segment_Week_3=d[d.Week == 3]
d_segment_Week_4=d[d.Week == 4]
d_segment_Week_5=d[d.Week == 5]
d_segment_Week_6=d[d.Week == 6]
d_segment_Week_7=d[d.Week == 7]
d_segment_Week_8=d[d.Week == 8]
d_segment_Week_9=d[d.Week == 9]
d_segment_Week_10=d[d.Week == 10]
d_segment_Week_11=d[d.Week == 11]
d_segment_Week_12=d[d.Week == 12]
print('----- d_segment_Week_12 [', index_d , '] ---------')
print('d_segment_Week_12 [', index_d , ']=', d_segment_Week_12[index_d])

#-------------------- ЕТАП ІІ підготовка даних для статистичної обробки ---------------------------
d_middle_Week= np.zeros(11)

def weekly_average (d_segment_Week, index_d):            # середне значення продажів за тиждень
    n = d_segment_Week[index_d].size                     # розмір стовпця
    d_segment_Week.index = range(0, len(d_segment_Week)) # балансування індексів в сегменті 0-n
    sum_Week=0
    for i in range(0, n):
        sum_Week = sum_Week + d_segment_Week[index_d][i]
    middle_Week=sum_Week/n
    return middle_Week

#-------------------- Головні виклики ІІ етапу ---------------------------
print('------------------- Middle_Week ------------------')
d_middle_Week[0]=weekly_average (d_segment_Week_2, index_d)        # середне значення продажів за тиждень
d_middle_Week[1]=weekly_average (d_segment_Week_3, index_d)
d_middle_Week[2]=weekly_average (d_segment_Week_4, index_d)
d_middle_Week[3]=weekly_average (d_segment_Week_5, index_d)
d_middle_Week[4]=weekly_average (d_segment_Week_6, index_d)
d_middle_Week[5]=weekly_average (d_segment_Week_7, index_d)
d_middle_Week[6]=weekly_average (d_segment_Week_8, index_d)
d_middle_Week[7]=weekly_average (d_segment_Week_9, index_d)
d_middle_Week[8]=weekly_average (d_segment_Week_10, index_d)
d_middle_Week[9]=weekly_average (d_segment_Week_11, index_d)
d_middle_Week[10]=weekly_average (d_segment_Week_12, index_d)
print('d_middle_Week=', d_middle_Week)
print('---------------------------------------------------')
plt.title('Middle_Week'); plt.plot(d_middle_Week)                   # візуалізація усередниних за тиждень даних
plt.show()

s1=pd.Series(d_middle_Week)
plt.title('Middle_Week')
s1.plot(kind='bar', color='b')
plt.show()


# 0.2 підготовка вхідних аних для оцінювання ефективності промо акцій -----------------------------
d_segment_Yes=d_segment[d_segment.Promo_action == 'так']             # сегментація за промо акцією "да"
d_segment_Yes.index = range(0, len(d_segment_Yes))                  # балансування індексів в сегменті 0-n
d_segment_Not=d_segment[d_segment.Promo_action == 'ні']            # сегментація за промо акцією "нет"
d_segment_Not.index = range(0, len(d_segment_Not))                  # балансування індексів в сегменті 0-n
print('---------------------------------------------------')
print('d_segment_Yes=', d_segment_Yes)
print('---------------------------------------------------')
print('d_segment_Not=', d_segment_Not)
plt.title('d_segment_Yes'); d_segment_Yes[index_d].plot()           # візуалізація усередниних за тиждень даних
plt.show()
plt.title('d_segment_Not'); d_segment_Not[index_d].plot()           # візуалізація усередниних за тиждень даних
plt.show()

#--------------------------- ЕТАП ІІІ статистична обробка ---------------------------

def Stst_A(S, Y_coord, title):                  # визначення статистичних хараткеристик
    iter = Y_coord.size
    S0 = np.zeros(iter)
    for i in range(iter):
        S0[i] = abs(S[i] - 0)  # урахування тренду в оцінках статистичних хараткеристик
    mS=np.mean(S0)
    dS=np.var(S0)
    scvS=mt.sqrt(dS)
    print('----- статистичны характеристики',title,'-----')
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    plt.title(title)
    plt.hist(S,  bins=20, facecolor="blue", alpha=0.5) # гістограма закону розподілу ВВ
    plt.show()
    return Stst_A


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

# -------------------------- МНК екстраполяція загальна -------------------------
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


# ------------------ МНК екстраполяція для промоакцій -----------------------
def MNK_Yes_Not_Extrapolation (Y_Yes, Y_Not , d_segment):
    # ---- формування структури вхідних матриць МНК ------
    # -------------- для проведених акцій -----------------
    iter = Y_Yes.size
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 5))
    for i in range(iter):
        Yin[i, 0] = Y_Yes[i, 0]
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
    j=d_segment.size
    Yout_Yes_E=np.zeros((j))
    for i in range(0, j):
        Yout_Yes_E[i] = C[0, 0] + C[1, 0]*i + (C[2, 0]*i*i) + (C[3, 0]*i*i*i) + (C[4, 0]*i*i*i*i)
    # ---- формування структури вхідних матриць МНК ------
    # -------------- для відсутніх акцій -----------------
    iter = Y_Not.size
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 5))
    for i in range(iter):
        Yin[i, 0] = Y_Not[i, 0]
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
        F[i, 3] = float(i * i * i)
        F[i, 4] = float(i * i * i * i)
    # ------------- алгоритм МНК ------------------------
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    j = d_segment.size
    Yout_Not_E=np.zeros((j))
    Y_Not_Yes = np.zeros((j))
    for i in range(0, j):
        Yout_Not_E[i] = C[0, 0] + C[1, 0] * i + (C[2, 0] * i * i) + (C[3, 0] * i * i * i) + (C[4, 0] * i * i * i * i)
    # ---- розрахунок показника ефективності акції ------
    for i in range(0, j):
        Y_Not_Yes[i]=Yout_Yes_E[i]-Yout_Not_E[i]

    return  Y_Not_Yes

#-------------------- Головні виклики ІІІ етапу ---------------------------

# 3.1 Статистична обробка динаміки продаж
n_OLS = len(d_segment[index_d])                                     # трансформатор DataFrame numpy
d_segment_OLS = np.zeros(n_OLS)
for i in range(0, n_OLS):
    d_segment_OLS[i] = d_segment[index_d][i]

# ---------------------------- ОЦІНЮВАННЯ ------------------------------
# ---------------- МНК для продажів за повною вибіркою -----------------
Yout0 = MNK (d_segment_OLS)
print('------------ вхідна вибірка  ----------')
Stst_A(d_segment_OLS, Yout0, 'вхідна вибірка повна')
print('-------------- МНК оцінка  ------------')
Stst_A(Yout0, Yout0, 'МНК оцінка за повною вибіркою')
plt.title('MNK_sale')
plt.plot(d_segment_OLS)
plt.plot(Yout0)
plt.show()

# ---------------------- МНК для продажів за тижнями -------------------
Yout1 = MNK (d_middle_Week)
print('------------ вхідна вибірка  ----------')
Stst_A(d_middle_Week, Yout1, 'вхідна вибірка за тижнями')
print('-------------- МНК оцінка  ------------')
Stst_A(Yout1, Yout1, 'МНК оцінка за тижнями')
plt.title('MNK_segment_week_sale')
plt.plot(d_middle_Week)
plt.plot(Yout1)
plt.show()

# ----------------------------- ПРОГНОЗ --------------------------------
# ---------------------- МНК для продажів за повною вибіркою -----------
prognoz=int (0.5*(n_OLS))
Graf_Yout0=np.zeros(prognoz)
Graf_Yout0_1=np.zeros(6)
Yout0 = MNK_extrapolation (d_segment_OLS, prognoz)
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

# ----------------- МНК для продажів за тижнями -------------------
prognoz=6
Graf_Yout1=np.zeros(prognoz)
Yout1 = MNK_extrapolation  (d_middle_Week, prognoz)
print('------ MNK_extrapolation_segment_week_sale --------')
for i in range(prognoz):
    print ('Yout1[', i, ',0]=', Yout1[i,0])
plt.title('MNK_extrapolation_segment_week_sale')
for i in range(0, prognoz):
    Graf_Yout1[i] = Yout1[i, 0]
plt.plot(Graf_Yout1)
plt.show()

# -------------------------------------------------------------------------
plt.title('MNK_extrapolation за повною вибіркою та тижнями')
plt.plot(Graf_Yout0_1)
plt.plot(Graf_Yout1)
plt.show()

#---- 3.1 Статистична обробка для оцінювання ефективності промо акції ------------------
#---- для наявної промо акції
n_Yes = len(d_segment_Yes[index_d])                                 # трансформатор DataFrame - numpy
d_segment_Yes_array = np.zeros(n_Yes)
for i in range(0, n_Yes):
    d_segment_Yes_array[i] = d_segment_Yes[index_d][i]

# ---------------------------- ОЦІНЮВАННЯ ------------------------------
# ---------------- МНК для продажів за d_segment_Yes -------------------
Yout_Yes = MNK ( d_segment_Yes_array)
print('------------ вхідна вибірка d_segment_Yes ----------')
Stst_A( d_segment_Yes_array, Yout_Yes, 'вхідна вибірка d_segment_Yes')
print('-------------- МНК оцінка Yes ------------')
Stst_A(Yout_Yes, Yout_Yes, 'МНК оцінка d_segment_Yes')
plt.title('MNK_d_segment_Yes')
plt.plot( d_segment_Yes_array)
plt.plot(Yout_Yes)
plt.show()

#---- за відсутної промо акції
n_Not = len(d_segment_Not[index_d])                                 # трансформатор DataFrame - numpy
d_segment_Not_array = np.zeros(n_Not)
for i in range(0, n_Not):
    d_segment_Not_array[i] = d_segment_Not[index_d][i]

# ---------------------------- ОЦІНЮВАННЯ ------------------------------
# ---------------- МНК для продажів за d_segment_Not -------------------
Yout_Not = MNK ( d_segment_Not_array)
print('------------ вхідна вибірка d_segment_Not ----------')
Stst_A( d_segment_Not_array, Yout_Not, 'вхідна вибірка d_segment_Not')
print('-------------- МНК оцінка Not ------------')
Stst_A(Yout_Not, Yout_Not, 'МНК оцінка d_segment_Not')
plt.title('MNK_d_segment_Not')
plt.plot( d_segment_Not_array)
plt.plot(Yout_Not)
plt.show()

# ---------------- МНК екстраполяція -------------------
Y_Not_Yes_efficiency = MNK_Yes_Not_Extrapolation (Yout_Yes, Yout_Not , d_segment)
plt.title('Y_Not_Yes_efficiency')
plt.plot(Y_Not_Yes_efficiency)
plt.show()
print('---------------- STOP  ---------------')
