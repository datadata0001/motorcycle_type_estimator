import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

#Veri hazirlama

ogrenim_hp = np.array([205,200,210,180,140,120,120,140,130,140,120,130,70,60,55,68,50,75,50,50,50])
ogrenim_tork = np.array([130,130,125,140,132,133,120,120,120,100,100,100,150,150,150,100,100,100,50,50,50])
ogrenim_kg = np.array([200,180,180,180,160,150,250,250,250,250,250,250,90,80,85,200,200,200,160,160,160])
ogrenim_topspeed = np.array([310,299,305,299,210,220,220,220,220,220,220,220,150,175,165,180,180,180,180,180,180])
ogrenim_depo = np.array([15,15,15,15,15,15,30,30,30,25,25,25,10,10,10,15,15,15,15,15,15])
ogrenim_silindir = np.array([4,4,4,2,3,2,2,2,2,2,2,2,1,1,1,2,4,2,1,1,1])
ogrenim_tur = ['Racing','Racing','Racing','Naked','Naked','Naked','Enduro','Enduro','Enduro','Touring','Touring','Touring','Motocross','Motocross','Motocross','Chopper','Chopper','Chopper','Scooter','Scooter','Scooter']
veri = pd.DataFrame({
    'HP': ogrenim_hp,
    'Tork': ogrenim_tork,
    'Kg': ogrenim_kg,
    'Top Speed': ogrenim_topspeed,
    'Depo': ogrenim_depo,
    'Silindir': ogrenim_silindir,
    'Tur': ogrenim_tur
})


print('Hangi Tip Motorsiklet Suruyorsunuz Bulalim. Hos Geldiniz!')
motor_hp = float(input('Kac beygir? : '))
motor_tork = float(input('Kac tork? : '))
motor_kg = float(input('Kac kg? : '))
motor_silindir = float(input('Kac silindir? : '))
top_speed = float(input('Son hizini giriniz. : '))
depo = float(input('Depo hacmini giriniz. : '))
tur = input('Motorunuzun ismini giriniz. : ')


motor_hp_ekleme = np.append(ogrenim_hp,[motor_hp])
motor_tork_ekleme = np.append(ogrenim_tork,[motor_tork])
motor_kg_ekleme = np.append(ogrenim_kg,[motor_kg])
motor_silindir_ekleme = np.append(ogrenim_silindir,[motor_silindir])
motor_top_speed = np.append(ogrenim_topspeed,[top_speed])
motor_depo = np.append(ogrenim_depo,[depo])
motor_tip_ekleme = np.append(ogrenim_tur,[tur])


veri_ekleme = pd.DataFrame({
    'HP':motor_hp_ekleme,
    'Tork':motor_tork_ekleme,
    'KG':motor_kg_ekleme,
    'Silindir':motor_silindir_ekleme,
    'TopSpeed':motor_top_speed,
    'Depo':motor_depo,
    'Tur': motor_tip_ekleme})

#TAHMIN ICIN

x = veri_ekleme.iloc[:,0:6]
y = veri_ekleme.iloc[:,-1:]
X = x.values



#LabelEncoder 
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
Y = le.fit_transform(y)

#Decision Tree
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

print('6 = Touring , 5 = Scooter , 4 = Racing , 3 = Naked , 2 = Motocross ,  1 = Enduro , 0 = Chopper')

print('DC Modeli')
tahmin_dc = print(r_dt.predict([[motor_hp,motor_tork,motor_kg,motor_silindir,top_speed,depo]]))

model = sm.OLS(r_dt.predict(X),X)
(model.fit().summary())

#Guncellenmis Decision Tree Denemesi

Xdt = np.append(arr =np.ones((22,1)).astype(int),values=X , axis = 1 )
X_dt = x.iloc[:,[1,3,4]].values
X_dt = np.array(X_dt,dtype=float)
model = sm.OLS(y, X_dt).fit() 
(model.summary())

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X_dt,Y)
print('Guncellenmis DC Modeli:')
tahmin_dc_guncellenmis =print(r_dt.predict([[motor_tork,motor_silindir,top_speed]]))

#Linear
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, Y)
Y_pred = regressor.predict(X)
print('Linear Modeli')
tahmin_lin = print(regressor.predict([[motor_hp,motor_tork,motor_kg,motor_silindir,top_speed,depo]]))

model = sm.OLS(regressor.predict(X),X)
(model.fit().summary())

#Guncellenmis Linear Denemesi.
Xlinear = np.append(arr =np.ones((22,1)).astype(int),values=X , axis = 1 )
X_l = x.iloc[:,[1,3,4]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(y, X_l).fit()
(model.summary())

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_l, Y)
Y_pred = regressor.predict(X_l)
print('Guncellenmis Linear Modeli:')
tahmin_lin_guncellenmis =print(regressor.predict([[motor_tork,motor_silindir,top_speed]]))


#****************************************************************************************************#

#SINIFLANDIRMA ICIN
t = veri.iloc[:,0:6]
z = veri.iloc[:,-1:]
T = t.values
Z = z.values

#train_test_split
from sklearn.model_selection import train_test_split
t_train , t_test , z_train , z_test = train_test_split(t,z,test_size=0.33,random_state=0)

#standard_Scaler
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

T_train = sc.fit_transform(t_train)
T_test = sc.transform(t_test)


from sklearn.decomposition import PCA 

pca = PCA(n_components = 2)

t_train2 = pca.fit_transform(T_train)
t_test2 = pca.transform(T_test)


#PCA LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(t_train2,z_train)

z_pred_pca = classifier.predict(t_test2)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(z_test, z_pred_pca)
print(cm)


#LDA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)

t_train_lda = lda.fit_transform(T_train , z_train)
t_test_lda = lda.transform(T_test)

from sklearn.linear_model import LogisticRegression
classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(t_train_lda,z_train)

z_pred_lda = classifier_lda.predict(t_test_lda)


from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(z_test,z_pred_lda)
print(cm2)




#Logistic Regresyon

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(T_train,z_train)

z_pred_logr = logr.predict(T_test)
print('Logistic Regresyon')
print(z_pred_logr)
print('Gercek Degerler')
print(z_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(z_test, z_pred_logr)
print(cm)

#knn 
from sklearn.neighbors import KNeighborsClassifier

knn= KNeighborsClassifier(n_neighbors=1,metric='euclidean')
knn.fit(T_train,z_train)

z_pred_knn = knn.predict(T_test)
print('KNN')
print(z_pred_knn)
print('Gercek Degerler')
print(z_test)

cm = confusion_matrix(z_test, z_pred_knn)
print(cm)


#svc
from sklearn.svm import SVC
svc = SVC(kernel = 'linear')
svc.fit(T_train,z_train)

z_pred_svc = svc.predict(T_test)
print('SVC')
print(z_pred_svc)
print('Gercek Degerler')
print(z_test)

cm = confusion_matrix(z_test,z_pred_svc)
print(cm)

#naiveBayes 
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(T_train,z_train)

z_pred_gnb = gnb.predict(T_test)
print('GNB')
print(z_pred_gnb)
print('Gercek Degerler')
print(z_test)

cm = confusion_matrix(z_test,z_pred_gnb)
print(cm)

#dc
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(T_train,z_train)

z_pred_dtc = dtc.predict(T_test)
print('DTC')
print(z_pred_dtc)
print('Gercek Degerler')
print(z_test)

cm = confusion_matrix(z_test,z_pred_dtc)
print(cm)

#RandomForest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=15,random_state=0)
rfc.fit(T_train,z_train)

z_pred_rfc = rfc.predict(T_test)
print('RFC')
print(z_pred_rfc)
print('Gercek Degerler')
print(z_test)

z_proba = rfc.predict_proba(T_test)

cm = confusion_matrix(z_test,z_pred_rfc)
print(cm)

#roc_auc

print(z_test)
print(z_proba[:,0])

from sklearn import metrics 
fpr , tpr , thold = metrics.roc_curve(z_test , z_proba[:,0],pos_label = 'Naked')
print(fpr)
print(tpr)

#****************************************************************************************************#

#KUMELEME ICIN

k = veri.iloc[:,0:2].values  #hp ve tork'u ele aldik ve bunlari kumeleyecek.

#K-Means 

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 4 , init ='k-means++')
kmeans.fit(k)

print(kmeans.cluster_centers_)

sonuclar = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state = 0)
    kmeans.fit(k)
    sonuclar.append(kmeans.inertia_)
    
plt.plot(range(1,11),sonuclar)
plt.show()

#K-Means Verilerin Dagilim Grafigi
kmeans = KMeans(n_clusters = 4 , init='k-means++',random_state = 0)
k_pred = kmeans.fit_predict(k)
print(k_pred)
plt.scatter(k[k_pred==0,0],k[k_pred==0,1],s=100,color='purple')
plt.scatter(k[k_pred==1,0],k[k_pred==1,1],s=100,color='yellow')
plt.scatter(k[k_pred==2,0],k[k_pred==2,1],s=100,color='pink')
plt.scatter(k[k_pred==3,0],k[k_pred==3,1],s=100,color='green')
plt.title('KMeans')
plt.show()

#HC

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 4 , affinity = 'euclidean' , linkage = 'ward')
k_pred = ac.fit_predict(k)
print(k_pred)

plt.scatter(k[k_pred==0,0],k[k_pred==0,1],s=100,color='purple')
plt.scatter(k[k_pred==1,0],k[k_pred==1,1],s=100,color='yellow')
plt.scatter(k[k_pred==2,0],k[k_pred==2,1],s=100,color='pink')
plt.scatter(k[k_pred==3,0],k[k_pred==3,1],s=100,color='green')
plt.title('HC')
plt.show()

#Dendogram 
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(k,method ='ward'))
plt.show()


#######################################################################################################################################
#Yapay Sinir Aglari 
t = veri.iloc[:,0:6]
z = veri.iloc[:,-1:]
T = t.values
Z = z.values


#Label & One-Hot Encoder
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
le = preprocessing.LabelEncoder()
Z = le.fit_transform(Z)
ohe = preprocessing.OneHotEncoder()
Z = ohe.fit_transform(veri.iloc[:,-1:]).toarray()

#Train_Test_Split
from sklearn.model_selection import train_test_split

t_train , t_test , z_train , z_test , = train_test_split(T,Z,test_size=0.33 , random_state=0 )

#Standard_Scaler

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

T_train = sc.fit_transform(t_train)
T_test = sc.transform(t_test)



#Yapay_Sinir_Aglari

import keras

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(42,kernel_initializer='uniform',activation = 'relu',input_dim = 6))

classifier.add(Dense(42,kernel_initializer='uniform',activation = 'relu'))

classifier.add(Dense(7, kernel_initializer ='uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(T_train,z_train , epochs = 420)

z_pred = classifier.predict(T_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(z_test, z_pred)
print(cm)







