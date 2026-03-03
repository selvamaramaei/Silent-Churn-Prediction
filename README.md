# 📊 Silent Churn Prediction & Monitoring System

##  Proje Hakkında (Executive Summary)

Bu proje, bir platformun kullanıcı tutundurma (retention) stratejisini güçlendirmek amacıyla, aboneliğini iptal etmeyen ancak kullanım sıklığı veya süresi gizlice azalan (**Silent Churn**) kullanıcıları tespit etmek için geliştirilmiştir.

Geleneksel "açık churn" modellerinden farklı olarak, bu sistem kullanıcıların davranışsal verilerindeki negatif trendleri (özellikle `usage_drop_rate`) analiz eder. Veri kümesindeki aşırı sınıf dengesizliğine (%1.84 churn oranı) rağmen, yüksek **Recall** odaklı mühendislik yaklaşımları kullanılarak, işletmeye müdahale şansı tanıyan "riskli kullanıcı listeleri" üretilir.

###  Çözülen Problem & İş Değeri
* **İş Problemi:** Kullanıcılar bir platformu kullanmayı bıraktıklarında genellikle aboneliklerini hemen iptal etmezler, bu da gelir kaybına yol açar.
* **Mühendislik Çözümü:** Bu sistem, ham kullanım günlüklerini işleyerek (features) yüksek riskli kullanıcıları (labels) belirler ve bu riskli kullanıcıların analizini sağlayan interaktif bir dashboard sunar.
* **Prioritize Edilen Metrik: RECALL (Sınıf 1).** Bir bilgisayar mühendisi olarak şu kararı verdim: "Kullanıcının sessizce gitmesini (False Negative) kaçırmaktansa, bazen sağlıklı müşteriye 'bir sorun mu var?' diye sormayı (False Positive) tercih ederim".

---

##  Proje Mimarisi & Veri Akışı (Architectural Diagram)

Proje, tam modüler ve ölçeklenebilir bir MLOps yapısına sahiptir. Ham verinin üretiminden son kullanıcının raporu görmesine kadar 4 ana katmandan oluşur.

```mermaid
graph TD
    %% -- Data Source & Ingestion --
    A[data_gen.py Script] -->|Ham Veri Üretimi| B[raw_usage_data.csv]
    B -->|Ingestion Flow| C(src/db_loader.py)
    
    %% -- Database Layer (PostgreSQL) --
    C -->|Ham Veri Yükleme| D[(PostgreSQL DB)]
    D -->|SQL Özellik Mühendisliği| D
    subgraph "Database Schema"
        raw_usage_data:::table
        processed_features:::table
        labeled_features:::table
    end
    
    %% -- Feature Pipeline --
    D -->|Features & Labels Çekme| E(src/feature_pipeline.py)
    E -->|Veri Ön İşleme & Labeling| E
    
    %% -- Model Training & MLOps --
    subgraph "Model Development"
        E -->|labeled_features| F(src/train_xgboots.py)
        E -->|labeled_features| G(src/train_random_forest.py)
        F -->|XGBoost Model| H(models/silent_churn_v1.json)
        G -->|Random Forest Model| I(models/rf_silent_churn_v1.joblib)
    end
    H & I -->|Tracking via .dvc| J(DVC Meta Files)
    
    %% -- Web Dashboard (Streaming UI) --
    subgraph "Inference & Visualization"
        D -->|processed_features| K(dashboard.py - Streamlit)
        H & I -->|Loading Models| K
        K -->|Risk Scoring (Inference)| K
        K -->|UI Components: User Lookup, Distribution, Risk Table| L[User Dashboard (Web UI)]
    end

    %% --- Styling ---
    classDef table fill:#e1f5fe,stroke:#01579b,stroke-width:1px,color:#01579b;
    classDef script fill:#f9fbe7,stroke:#33691e,stroke-width:1px;
    classDef output fill:#ffecb3,stroke:#ff8f00,stroke-width:1px;
    classDef ui fill:#e0f2f1,stroke:#00695c,stroke-width:2px;
    
    linkStyle default stroke:#666,stroke-width:1px,fill:none;
```

 
