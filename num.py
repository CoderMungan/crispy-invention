import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



yolcu_df = pd.read_csv("titanic.csv") #Datamızı okuma komudu ekliyoruz. Bundan sonra yolcu_df değeri sürekli data okuyacaktır.



yolcu_df.info() #Data hakkında bilgi



yolcu_df.describe() #Tablonun İstatistik Bilgisi



yolcu_df=yolcu_df.drop(['PassengerId','Name','Sex'],axis=1) #Yolcunun Adı, SexCodu, ID si çıkarıldı. 'drop' fonksiyonu ile



yolcu_df.head() #Çıkarıldıktan Sonra İlk 5 Satır Görüntülendi



erkek_yolcu = yolcu_df[yolcu_df['Sex']== 'male']  #Erkek_yolcu değeri atandı

kadın_yolcu = yolcu_df[yolcu_df['Sex']== 'female'] #Kadın_yolcu değeri atandı



cocuk_yolcu = yolcu_df[yolcu_df['Age'] < 16] #Çocuk_yolcu Değeri Atandı

erkek_cocuk_yolcu = cocuk_yolcu[cocuk_yolcu['Sex'] == 'male'] #Erkek_Çocuk_Yolcu Değeri Atandı

kadın_cocuk_yolcu = cocuk_yolcu[cocuk_yolcu['Sex'] == 'female'] #Kadın_Çocuk_Yolcu Değeri atandı



yetiskin_erkek_yolcu = erkek_yolcu.drop(erkek_cocuk_yolcu.index[:]) # Yetişkin erkek yolcu öğrenilsin diye çocuk erkek yolcular çıkarıldı.

yetiskin_kadın_yolcu = kadın_yolcu.drop(kadın_cocuk_yolcu.index[:]) # Yetişkin erkek yolcu öğrenilsin diye çocuk erkek yolcular çıkarıldı.



x = [len(erkek_yolcu), len(kadın_yolcu)] #İlk tablo

label = ['Male', 'Female']

plt.pie(x, labels = label, autopct = '%1.01f%%')

plt.title('Yolcuların Cinsiyet Grafiği')

plt.show()



def yas_problemi(x): # Bu fonksiyonda cocuk genç yetişkin olarak sınıflandırma yapıldı

    if x>=0 and x <16:

        return 'Cocuk'

    elif x>=16 and x<=24:

        return 'Genc'

    else:

        return 'Yetiskin'

       

yolcu_df['Age'].apply(yas_problemi).value_counts() # Burada Tam Anlamıyla Yaş Aralıkları Belli Olmuştur



yolcu_df['Age'].apply(yas_problemi).value_counts().plot(kind='pie', autopct='%1.0f%%') # Belli olan yaş grafiği ile grafik olarak çıktı alıyoruz

plt.title('Yolcuların Yas Ortalaması')

plt.show()



print ('Yetişkin erkek yolcuların yaş ortalaması:', yetiskin_erkek_yolcu['Age'].mean()) #Yaş Ortalaması Alınmıştır

print ('Yetişkin kadın yolcuların yaş ortalaması:', yetiskin_kadin_yolcu['Age'].mean())

print ('Çocuk yolcuların yaş ortalaması:', cocuk_yolcu['Age'].mean())



yolcu_df['Pclass'].value_counts() #Yolcuların Hangi Sınıflardan Oluştuğu görüntülenmiştir.



yolcu_df['Pclass'].value_counts().plot(kind='barh', color='green', figsize=[16,4]) #Yolcuların Hangi Sınıflardan Oluştuğunun Grafik Çıktısıdır.

plt.xlabel('Sıklık')

plt.ylabel('Yolcu sınıfı')

plt.show()



yolcu_df['Survived'].value_counts() # Yaşayan ve ölen yolcuların sayısı



yolcu_df['Survived'].value_counts().plot(kind='bar', title='Hayatta Kalma') # Yaşayan ve ölen yolcuların grafik çıktısı

plt.xlabel('0= Ölü  1= Hayatta')

plt.ylabel('Sıklık')

plt.show()



yolcu_df.groupby('Sex')['Survived'].value_counts() # Yaşayan ve ölen yolcuların cinsiyetine göre değerleri



yolcu_df.groupby('Sex')['Survived'].value_counts().plot(kind='bar', stacked=True, colormap='winter') # Yaşayan ve ölen yolcuların cinsiyetine göre değerleri grafik çıktısı

plt.show()



cinsiyet_hayattakalma = yolcu_df.groupby(['Sex', 'Survived']) # Yaşayan ve ölen yolcuların cinsiyetine göre değerleri grafik çıktısı farklı version

cinsiyet_hayattakalma.size().unstack().plot(kind='bar', stacked=True, colormap='winter')

plt.ylabel('Sıklık')

plt.title('Cinsiyete Göre Hayatta Kalma')

plt.show()



print ('Hayatta kalan yetiskin kadın yolcular:', yetiskin_kadin_yolcu['Survived'].mean())

print ('Hayatta kalan yetiskin erkek yolcular:', yetiskin_erkek_yolcu['Survived'].mean())



sınıf_hayattakalma = yolcu_df.groupby(['Pclass', 'Survived'])



sınıf_hayattakalma.size()



sınıf_hayattakalma.size().unstack()



sınıf_hayattakalma.size().unstack().plot(kind='bar', stacked=True, colormap='autumn')

plt.xlabel('1st = Üst Seviye,   2nd = Orta Seviye,   3rd = Düşük Seviye')

plt.ylabel('Sıklık')

plt.title('Ölen veya Yasayan Yolcuların Sınıfları')

plt.show()
