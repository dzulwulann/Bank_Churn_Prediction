# Churn Prediction for Bank Customer 
Creating a Machine Learning model to predict Customer who has the potential to churn. 

## Stage 0 
**Problem Statement**

Perusahaan Rakamin Bank Center (RBC) tidak memiliki model Machine Learning (ML) untuk memprediksi nasabah mana yang akan <br>
churn. Dari data historikal yang ada, diperoleh jumlah nasabah churn sebesar **20,37%** dari keseluruhan data. Mengacu pada laman <br>
uxpressia.com tentang "*How to Approach Customer Churn Measurements in Banking*", toleransi nasabah churn maksimal sebesar <br>
**10%**. Sementara itu, jumlah nasabah churn pada data yang kita miliki melebihi batas toleransi tersebut. Dengan model ML yang dibuat, <br>
diharapkan menjadi acuan bagi tim bisnis untuk mengambil langkah strategi mengatasi nasabah yang terdeteksi churn. 

----------
**Goals** 

Membuat model Machine Learning dengan tingkat akurasi > () dan tingkat presisi > () untuk membantu bank Rakamin Bank Center (RBC) 
dalam memprediksi nasabah yang akan churn dan membantu tim bisnis dalam menentukan strategi terhadap nasabah yang akan churn

---------
Objectives 

* Mengidentifikasi variabel yang memiliki relevansi dengan keputusan nasabah untuk berhenti berlangganan
* Mempersiapkan data historikal yang digunakan untuk model Machine Learning
* Membangun model prediktif untuk mengklasifikasikan nasabah yang berpotensi churn
* Melakukan optimasi model sehingga mendapatkan hasil yang terbaik

----------
Bussines Metrics

Untuk mengukur keberhasilan objective tersebut dengan **churn rate**. Kami juga mengukur metric sekunder dengan tingkat **akurasi model**.

## Stage 1

**EDA**

Kami menggunakan data set bank customers yang berasal dari [kaggle.com](https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers). Berikut beberapa insight yang kami temukan 

<img src="./image/data_info.png" alt="drawing" width="300"/>

Dataset berisi 10.000 row data dengan 14 fitur kolom. Kolom Exited sebagai variable target. Di dalam data tidak ada *missing value* untuk semua atribut.

<img src="./image/data_describe_num.png" alt="decribe" />

Jarak antara nilai median dan rata - rata CreditScore, Age, EstimatedSalary, dan Tenure  sangat dekat, sehingga berdasarkan angkanya, sebaran data cenderung memiliki distribusi normal. Sedangkan untuk fiture balance, jarak antara nilai median dan rata - rata berjauhan sehingga cenderung memiliki distibusi skew dan memiliki nilai outlier yang ekstrim.

<img src="./image/data_describe_categorical.png" alt="decribe" />

Fitur Geography memiliki 3 nilai unik dengan dominasi negara France (50%). Fitur Gender dengan 2 nilai unik. Fitur HasCard dan IsActiveMember dengan 2 nilai unik. Sedangkan variable target Exited memiliki 2 nilai unik dengan jumlah nilai (1) atau nasabah sudah tidak menggunakan jasa bank lagi sebesar 21%.





