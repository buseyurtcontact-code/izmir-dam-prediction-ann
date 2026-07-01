flowchart TD
    A[Başla] --> B[Veriyi Yükle]

    B --> C[Veri Temizleme]
    C --> C1[Eksik yağış → 0]
    C --> C2[Eksik buharlaşma → 0]
    C --> C3[%100 üzerindeki dolulukları temizle]

    C --> D[Feature Engineering]

    D --> D1[Weather Features]
    D1 --> D11[7 günlük yağış]
    D1 --> D12[30 günlük yağış]
    D1 --> D13[90 günlük yağış]
    D1 --> D14[7 günlük sıcaklık ortalaması]

    D --> D2[Lag Features]
    D2 --> D21[Lag7]
    D2 --> D22[Lag14]
    D2 --> D23[Lag30]
    D2 --> D24[Lag60]
    D2 --> D25[Lag90]

    D --> D3[Trend Features]
    D --> D4[Seasonal Features]

    D --> E[Target (+7, +14, +30 gün)]

    E --> F[Eğitim / Test Ayrımı]
    F --> G[MinMax Scaling]
    G --> H[Sequence Oluşturma]

    H --> I[Model Kurulumu]

    I --> I1[Dense]
    I --> I2[LSTM]
    I --> I3[GRU]

    I --> J[Early Stopping ile Eğitim]
    J --> K[Tahmin Yap]
    K --> L[Tahminleri Geri Ölçekle]
    L --> M[Fiziksel Sınır (0–100)]

    M --> N[Performans Hesaplama]
    N --> N1[RMSE]
    N --> N2[MAE]
    N --> N3[R²]

    N --> O[Grafikler]
    O --> P[CSV Kaydet]
    P --> Q[Bitir]
