# 📊 Silent Churn Prediction & Monitoring System

##  Proje Hakkında 

Bu proje, bir platformun kullanıcı tutundurma (retention) stratejisini güçlendirmek amacıyla, aboneliğini iptal etmeyen ancak kullanım sıklığı veya süresi gizlice azalan (**Silent Churn**) kullanıcıları tespit etmek için geliştirilmiştir.

Geleneksel "açık churn" modellerinden farklı olarak, bu sistem kullanıcıların davranışsal verilerindeki negatif trendleri (özellikle `usage_drop_rate`) analiz eder. Veri kümesindeki aşırı sınıf dengesizliğine (%1.84 churn oranı) rağmen, yüksek **Recall** odaklı mühendislik yaklaşımları kullanılarak, işletmeye müdahale şansı tanıyan "riskli kullanıcı listeleri" üretilir.

###  Çözülen Problem & İş Değeri
* **İş Problemi:** Kullanıcılar bir platformu kullanmayı bıraktıklarında genellikle aboneliklerini hemen iptal etmezler, bu da gelir kaybına yol açar.
* **Mühendislik Çözümü:** Bu sistem, ham kullanım günlüklerini işleyerek (features) yüksek riskli kullanıcıları (labels) belirler ve bu riskli kullanıcıların analizini sağlayan interaktif bir dashboard sunar.
* **Prioritize Edilen Metrik: RECALL (Sınıf 1).** Bir bilgisayar mühendisi olarak şu kararı verdim: "Kullanıcının sessizce gitmesini (False Negative) kaçırmaktansa, bazen sağlıklı müşteriye 'bir sorun mu var?' diye sormayı (False Positive) tercih ederim".

---

##  Proje Mimarisi & Veri Akışı 

Proje, tam modüler ve ölçeklenebilir bir MLOps yapısına sahiptir. Ham verinin üretiminden son kullanıcının raporu görmesine kadar 4 ana katmandan oluşur.

```text
[ VERİ ÜRETİMİ ] --> [ VERİTABANI ] --> [ ÖZELLİK MÜHENDİSLİĞİ ] --> [ MODELLEME ] --> [ İZLEME ]
      |                  |                      |                       |               |
 data_generation.py  PostgreSQL DB       feature_pipeline.py      XGBoost & RF     dashboard.py
 (Ham CSV Üretimi)   (SQL Depolama)      (7d/30d Ortalamalar)     (Eğitim & DVC)   (Risk Analizi)
   
```

## Klasör Yapısı

```text
 Silent-Churn/
├── data/                    # Ham ve işlenmiş veriler (.dvc ile takip edilir)
│   ├── processed/           # SQL ingestion sonrası meta veriler
│   ├── quarantine/          # Kalite kontrolünden geçemeyen hatalı veriler
│   └── raw/                 # feature_usage.csv ve ham loglar
├── models/                  # Eğitilmiş model ağırlıkları
│   ├── rf_silent_churn_v1.joblib  # Random Forest Modeli
│   ├── silent_churn_v1.json       # XGBoost Modeli
│   └── *.dvc                # Büyük modellerin versiyon takibi
├── notebooks/               # Analiz ve EDA çalışmaları
│   ├── EDA.ipynb            # Keşifçi Veri Analizi
│   └── Model_Analysis.ipynb # Model Performans Derin Analizi
├── src/                     # Kaynak Kodlar (Logic Layer)
│   ├── databases/           # DB Bağlantı: db_loader.py
│   ├── features/            # Feature Pipeline: feature_pipeline.py
│   ├── flows/               # Ingestion Flow: ingestion_flow.py
│   ├── ingestion/           # Üretim: data_generation.py, data_quality.py
│   ├── labeling/            # Churn Tanımlama: labeler.py
│   ├── training/            # Eğitim: train_xgboots.py, train_random_forest.py
│   ├── dashboard.py         # Interaktif Streamlit Dashboard
│   ├── inference.py         # XGBoost Tahminleme Scripti
│   └── inference_rf.py      # Random Forest Tahminleme Scripti
└── .gitignore               # venv ve cache dosyalarının filtrelenmesi
```

## Model Performans Analizi
Model eğitiminde aşırı sınıf dengesizliği (%1.84 churn oranı) nedeniyle, modellerin 1 (churn) sınıfına daha fazla ağırlık vermesi (scale_pos_weight/class_weight) sağlanmıştır.

```text
| Metrik (Sınıf 1 - Churn) | XGBoost (Threshold 0.85) | Random Forest (Threshold 0.50) |
|--------------------------|--------------------------|-------------------------------|
| Recall (Duyarlılık)      | 0.71                     | 0.98                          |
| Precision (Kesinlik)      | 0.18                     | 0.12                          |
| F1-Score                 | 0.28                     | 0.21                          |
```

## Veriseti Mimarisi ve PostgreSQL Entegrasyonu
Projenin veri katmanı, davranışsal churn sinyallerini yakalayabilmek için tasarlanmış yapısal bir kullanım log mimarisine dayanır.

### Raw Dataset Schema (`feature_usage.csv`)

Her satır bir kullanıcının belirli bir feature ile etkileşimini temsil eder:

| Alan | Açıklama |
|------|----------|
| usage_id | Benzersiz kullanım kaydı kimliği |
| subscription_id | Abonelik kimliği |
| usage_date | Kullanım tarihi |
| feature_name | Kullanılan özellik |
| usage_count | Günlük kullanım sayısı |
| usage_duration_secs | Toplam kullanım süresi (saniye) |
| error_count | Oluşan hata sayısı |

Bu yapı, kullanıcı davranışını zaman serisi perspektifinden analiz edebilmek için normalize edilmiştir.

---

### PostgreSQL Feature Engineering Katmanı

Ham veriler PostgreSQL veri tabanına ingest edildikten sonra, abonelik bazlı davranışsal metrikler hesaplanır. Bu katman modelleme sürecinden önce veri zenginleştirme ve özetleme işlemlerini gerçekleştirir.

Hesaplanan temel özellikler:

- **daily_usage**  
  Aboneliğin ilgili gündeki toplam kullanım hacmi.

- **usage_7d_avg**  
  Son 7 günün hareketli ortalama kullanım değeri (kısa vadeli trend göstergesi).

- **usage_30d_avg**  
  Son 30 günün hareketli ortalama kullanım değeri (uzun vadeli davranış baz çizgisi).

- **usage_drop_rate**  
  `usage_7d_avg / usage_30d_avg` oranı.  
  Bu oran düştükçe kullanıcının platformdan davranışsal olarak uzaklaşma riski artar.

Bu yaklaşım, açık churn (abonelik iptali) yerine kullanım davranışındaki düşüşü analiz ederek erken risk tespiti yapmayı hedefler.

---

##  MLOps: DVC (Data Version Control)

Bu projede büyük veri dosyaları ve model ağırlıkları doğrudan Git deposunda tutulmamaktadır. Bunun yerine **DVC (Data Version Control)** kullanılarak versiyonlanmaktadır.

### Versiyonlama Stratejisi

- Her model eğitimi sonrası:
  - Kullanılan veri seti snapshot olarak kaydedilir
  - Eğitilen model dosyası DVC ile takip altına alınır
  - `.dvc` referans dosyası oluşturulur

- GitHub repository:
  - Yalnızca referans dosyalarını içerir
  - Büyük veri ve model dosyaları DVC storage üzerinde saklanır

### Sağlanan Avantajlar

- Reproducible model eğitimi
- Deneylerin karşılaştırılabilirliği
- Hafif ve temiz Git repository yapısı
- Production ortamına uygun veri yönetimi

---
##  Kurulum

```bash
git clone https://github.com/username/Silent-Churn.git
cd Silent-Churn
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Model Eğitimi

```bash
python src/training/train_xgboots.py
```

---

## 📊 Interactive Dashboard (Streamlit)

Model çıktılarının analiz edilebilmesi için interaktif bir Streamlit dashboard geliştirilmiştir.

Dashboard üzerinde:

- **Model seçimi** (XGBoost / Random Forest)
- **Risk threshold ayarı**
- **Analiz edilecek kullanıcı sayısı belirleme**
- **Risk score dağılımı histogramı (log scale)**
- **High-risk kullanıcı listesi**
- **Account ID ile bireysel kullanıcı arama**
- **Kullanıcı bazlı usage_drop_rate trend grafiği**

Dashboard, global risk görünümü ile mikro kullanıcı analizi arasında geçiş yapmayı sağlar.

---
