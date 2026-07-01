import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import json
import math
import os
import random
import warnings

# Grafik üretimi için arka plan motorunu (backend) ayarlar
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# Model performans değerlendirme metrikleri ve veri ölçekleme kütüphaneleri
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

# Sabit Değerlerin Tanımlanması (Global Konfigürasyon)
RANDOM_SEED = 1  # Sonuçların tekrarlanabilir olması için rastgelelik tohumu
ALL_DAMS = ["Guzelhisar", "Balcova", "Gordes", "Tahtali", "Urkmez", "AlacatiKutluAktas"]  # Analiz edilecek barajlar
WEATHER_PREFIXES = ["tmax_", "tmin_", "tavg_", "hum_", "wind_", "sun_", "evap_", "prcp_"]  # Meteorolojik değişken ön ekleri
DEFAULT_LAG_DAYS = [7, 14, 30, 60, 90]  # Varsayılan gecikme (lag) günleri
FEATURE_CONFIG_PATH = "feature_config.json"  # Baraj bazlı öznitelik ayar dosyası
OUT_DIR = "results_test2023"  # Sonuçların ve grafiklerin kaydedileceği klasör
TARGET_HORIZONS = [7, 14, 30]  # Tahmin ufukları (Kaç gün sonrası tahmin edilecek?)
DAM_END_DATES = {"Guzelhisar": pd.Timestamp("2025-11-19")}  # Güzelhisar barajı için özel bitiş tarihi
_warned = set()  # Aynı uyarının birden fazla ekrana basılmasını engellemek için küme


def warn_once(key, msg):
    """Aynı uyarının konsolda yalnızca bir kez gösterilmesini sağlayan yardımcı fonksiyon."""
    if key not in _warned:
        _warned.add(key)
        warnings.warn(msg)


def parse_arguments():
    """Terminalden (Command Line) çalıştırılırken baraj ve başlangıç yılı parametrelerini alan fonksiyon."""
    parser = argparse.ArgumentParser(description="Train dense models for dam fill level prediction.")
    parser.add_argument(
        "--dams",
        type=str,
        default=None,
        help="Virgülle ayrılmış baraj listesi. Örnek: --dams \"Urkmez,Tahtali\"",
    )
    parser.add_argument(
        "--startyear",
        type=int,
        default=None,
        help="Eğitim verisine dahil edilecek en erken yıl. Örnek: --startyear 2015",
    )
    return parser.parse_args()


def set_random_seed(seed=RANDOM_SEED):
    """Deneylerin tekrarlanabilirliğini sağlamak amacıyla tüm kütüphanelerin rastgelelik tohumlarını sabitler."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def resolve_dams(dams_argument):
    """Kullanıcının seçtiği barajları kontrol eder, geçersiz olanları ayıklar ve analiz listesini döner."""
    if not dams_argument:
        return ALL_DAMS

    requested = [d.strip() for d in dams_argument.split(",")]
    invalid = [d for d in requested if d not in ALL_DAMS]

    if invalid:
        print(f"[WARNING] Bilinmeyen baraj(lar) yok sayıldı: {invalid}")

    DAMS = [d for d in requested if d in ALL_DAMS]

    if not DAMS:
        print("[ERROR] Geçerli bir baraj belirtilmedi. Program sonlandırılıyor.")
        exit(1)

    return DAMS


def make_sample_weights(index, halflife_days=365):
    """
    Tezde bahsedilen 'Örneklem Ağırlıklandırma' işlemi.
    Yakın tarihteki verilere üstel azalma (exponential decay) mantığıyla daha yüksek ağırlık verir.
    """
    latest = index.max()
    days_ago = np.array((latest - index).days, dtype=float)
    weights = np.exp(-days_ago * math.log(2) / halflife_days)  # Yarılanma ömrüne göre ağırlık hesabı
    return weights / weights.mean()  # Ağırlıkları normalize eder


def load_feature_config(path):
    """Her barajın hangi istasyon verilerini ve üretim kolonlarını kullanacağını belirten JSON dosyasını yükler."""
    with open(path, "r") as f:
        feature_config = json.load(f)

    print(f"Baraj öznitelik konfigürasyonu yüklendi: {list(feature_config.keys())}\n")
    return feature_config


def load_raw_data(path="son_merged.csv"):
    """Ham veri setini (CSV) yükler, tarih kolonunu indeks yapar ve kronolojik sıraya dizer."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("date").set_index("date")
    df = df.drop(columns=["Date"], errors="ignore")
    df = df.dropna(axis=1, how="all")  # Tamamen boş kolonları temizler
    return df


def clean_raw_data(df):
    """Veri temizleme aşaması: Eksik yağış/buharlaşma verilerini 0 ile doldurur ve %100'den büyük hatalı dolulukları ayıklar."""
    df = df.copy()
    prcp_all = [c for c in df.columns if c.startswith("prcp_")]
    evap_all = [c for c in df.columns if c.startswith("evap_")]

    # Yağış ve buharlaşma eksik verileri için 'Eksik Veri Göstergesi' oluşturma ve 0 atama
    for col in prcp_all + evap_all:
        df[f"{col}_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(0)

    print(f"Ham Veri Boyutu: {len(df)} satır, {len(df.columns)} sütun")
    print(f"Tarih Aralığı: {df.index.min().date()} -> {df.index.max().date()}\n")

    doluluk_cols = [c for c in df.columns if "DolulukOrani" in c]

    # %100'ün üzerindeki hatalı kayıtları mantıksal olarak NaN (boş veri) yapar
    for col in doluluk_cols:
        n_bad = (df[col] > 100).sum()
        if n_bad > 0:
            print(f"  [{col}] %100'den büyük olan {n_bad} adet hatalı kayıt temizleniyor.")
        df.loc[df[col] > 100, col] = np.nan

    print()
    return df


def get_dam_config(dam, cfg_override=None):
    """İlgili baraja ait model tipi, dizi uzunluğu ve strateji ayarlarını çeker."""
    cfg = cfg_override if cfg_override is not None else FEATURE_CONFIG[dam]
    return {
        "cfg": cfg,
        "prod_cols": cfg.get("prod_cols", []),
        "weather_stations": cfg.get("weather_stations", []),
        "lag_days": cfg.get("lag_days", DEFAULT_LAG_DAYS),
        "model_type": cfg.get("model", "GRU"),  # Varsayılan model GRU
        "seq_lens": cfg.get("seq_lens", [30, 60, 90]),  # +7, +14, +30 ufukları için geçmiş pencere uzunlukları
        "halflives": cfg.get("halflives", [365]),
        "strategy_name": cfg.get("strategy", "baseline"),
    }


def make_horizon_config(lag_days, seq_lens):
    """Her bir tahmin ufku (+7, +14, +30) için kullanılacak dizi uzunluğunu (seq_len) eşleştirir."""
    if len(seq_lens) != len(TARGET_HORIZONS):
        warn_once(
            "seq_lens_length",
            "seq_lens uzunluğu TARGET_HORIZONS uzunluğuyla aynı değil. Bazı tahmin ufukları eksik kalabilir.",
        )

    return {
        h: {"lags": list(lag_days), "seq_len": sl}
        for h, sl in zip(TARGET_HORIZONS, seq_lens)
    }


def select_dam_columns(dam, prod_cols, weather_stations):
    """Barajın hedef doluluk oranı kolonunu ve ilgili meteoroloji istasyonlarının verilerini seçer."""
    target_col = f"{dam}DolulukOrani"
    keep = [target_col] + prod_cols

    for station in weather_stations:
        for prefix in WEATHER_PREFIXES:
            col = f"{prefix}{station}"
            if col in df_raw.columns:
                keep.append(col)

    keep = [c for c in keep if c in df_raw.columns]
    missing = [c for c in prod_cols if c not in df_raw.columns]

    if missing:
        print(f"  [WARNING] Eksik kolonlar (atlandı): {missing}")

    return target_col, keep


def add_weather_features(df, weather_stations):
    """Öznitelik Mühendisliği: Yağışlar için 7, 30, 90 günlük hareketli toplamları ve 7 günlük sıcaklık ortalamasını hesaplar."""
    for station in weather_stations:
        prcp_col = f"prcp_{station}"
        if prcp_col in df.columns:
            df[f"rain_7d_{station}"] = df[prcp_col].rolling("7D").sum()
            df[f"rain_30d_{station}"] = df[prcp_col].rolling("30D").sum()
            df[f"rain_90d_{station}"] = df[prcp_col].rolling("90D").sum()

    for station in weather_stations:
        tavg_col = f"tavg_{station}"
        if tavg_col in df.columns:
            df[f"temp_7d_{station}"] = df[tavg_col].rolling("7D").mean()

    return df


def add_lag_and_trend_features(df, target_col, lag_days, weather_stations):
    """Öznitelik Mühendisliği: Geçmiş doluluk oranlarını (Lag) ve değişim hızlarını (Delta, Hareketli Ortalama) türetir."""
    # Gecikme (Lag) değerlerinin eklenmesi
    for lag in lag_days:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    # Anlık değişim hızlarını yakalayan Delta değişkenleri
    base_lag = min(lag_days)
    df["delta_1"] = df[target_col].diff(1).shift(base_lag)

    if 7 in lag_days:
        df["delta_7"] = df[target_col].diff(7).shift(base_lag)

    if max(lag_days) > 30 and 30 in lag_days:
        df["delta_30"] = df[target_col].diff(30).shift(base_lag)

    # Kısa ve uzun dönemli trend takibi için hareketli istatistikler ve farklar (trend_direction)
    df["rolling_mean_30"] = df[target_col].rolling(30).mean().shift(1)
    df["rolling_std_30"] = df[target_col].rolling(30).std().shift(1)
    df["rolling_mean_180"] = df[target_col].rolling(180).mean().shift(1)
    df["rolling_mean_365"] = df[target_col].rolling(365).mean().shift(1)
    df["trend_direction"] = df["rolling_mean_180"] - df["rolling_mean_365"]

    # Etkileşim Özelliği: Yağış miktarı ile barajın geçmiş doluluk düzeyinin çarpımı
    for station in weather_stations:
        prcp_col = f"prcp_{station}"
        if prcp_col in df.columns:
            df[f"rain_x_level_{station}"] = df[prcp_col] * df[f"lag_{base_lag}"]

    return df


def add_seasonal_features(df):
    """Öznitelik Mühendisliği: Takvim etkilerini / mevsimselliği sinüs ve kosinüs dönüşümleriyle doğrusal olmayan ağa aktarır."""
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    return df


def build_feature_dataframe(dam, prod_cols, weather_stations, lag_days):
    """Tüm öznitelik mühendisliği fonksiyonlarını çağırarak nihai veri tablosunu hazırlar, hedef kolonları kaydırır (shift)."""
    target_col, keep = select_dam_columns(dam, prod_cols, weather_stations)
    df = df_raw[keep].copy()

    for col in prod_cols:
        if col in df.columns:
            df[col] = df[col].shift(1)  # Gelecek verisinin sızmasını önlemek için üretim/talep verisini 1 gün kaydırır

    # Özellik mühendisliği adımlarının uygulanması
    df = add_weather_features(df, weather_stations)
    df = add_lag_and_trend_features(df, target_col, lag_days, weather_stations)
    df = add_seasonal_features(df)

    # Eksik verileri ileriye doğru doldurma (Forward Fill)
    feature_cols = df.columns.tolist()
    df[feature_cols] = df[feature_cols].ffill()

    # Çoklu tahmin ufku için hedef kolonların (+7, +14, +30 gün sonrasının verisi) oluşturulması
    target_cols_multi = []
    for h in TARGET_HORIZONS:
        col = f"target_{h}"
        df[col] = df[target_col].shift(-h)  # Gelecekteki doluluk oranını bugüne hedef değişken olarak getirir
        target_cols_multi.append(col)

    df = df.dropna(subset=target_cols_multi)
    df = df.dropna()

    return df, target_col


def determine_dam_start(dam, target_col, startyear):
    """Barajın verisinin başladığı doğal tarihi bulur veya kullanıcının belirttiği başlangıç yılıyla sınırlar."""
    dam_start = df_raw[target_col].first_valid_index()

    if dam_start is None:
        print(f"  [WARNING] {dam} için geçerli doluluk verisi bulunamadı, atlanıyor.\n")
        return None

    dam_start = pd.Timestamp(dam_start)
    selected_startyear = startyear if startyear is not None else (args.startyear if hasattr(args, "startyear") else None)

    if selected_startyear is not None:
        requested_start = pd.Timestamp(f"{selected_startyear}-01-01")
        if requested_start < dam_start:
            print(f"  [INFO] startyear {selected_startyear}, verinin başlangıcından önce; doğal başlangıç kullanılacak.")
        else:
            dam_start = requested_start

    return dam_start


def limit_dam_date_range(df, dam, target_col, startyear):
    """Veri setini belirlenen başlangıç tarihi ve baraj nihai bitiş tarihine göre filtreler."""
    dam_start = determine_dam_start(dam, target_col, startyear)

    if dam_start is None:
        return None

    dam_end = DAM_END_DATES.get(dam, pd.Timestamp("2025-11-30"))
    return df[(df.index >= dam_start) & (df.index <= dam_end)]


def split_train_test(df, target_col):
    """Tezde belirtilen sabit test ayrım tarihi olan 2023-01-01'e göre veri setini böler."""
    split_date = pd.Timestamp("2023-01-01")
    test_data = df[df.index >= split_date].copy()

    if target_col in test_data.columns:
        test_data[target_col] = test_data[target_col].ffill()
        test_data = test_data.dropna(subset=[target_col])

    # Kronolojik yapıyı bozmamak için her tahmin ufkuna göre geçmiş eğitim setini ayarlar
    train_per_horizon = {}
    for h in TARGET_HORIZONS:
        target_date_h = df.index + pd.Timedelta(days=h)
        train_h = df[target_date_h < split_date].copy()
        if target_col in train_h.columns:
            train_h[target_col] = train_h[target_col].interpolate(method="time")  # Zaman bazlı doğrusal interpolasyon
        train_per_horizon[h] = train_h

    return train_per_horizon, test_data


def make_dynamic_wf_windows(full_df, start_date, months_step):
    """Tez Bölüm 3.5.3'te geçen 'Genişleyen Pencereli İleri Yürüyüş Stratejisi' (Walk-Forward Expanding) pencerelerini üretir."""
    windows = []
    current_test_start = pd.Timestamp(start_date)
    max_date = full_df.index.max()

    while current_test_start < max_date:
        current_test_end = current_test_start + pd.DateOffset(months=months_step)  # 6 aylık adımlarla genişler
        train_slice = full_df[full_df.index < current_test_start]  # Test penceresinden önceki tüm geçmiş eğitim setidir
        test_slice = full_df[(full_df.index >= current_test_start) & (full_df.index < current_test_end)]

        if len(test_slice) > 0 and len(train_slice) > 60:
            windows.append((train_slice, test_slice))

        current_test_start = current_test_end

    return windows


def make_strategies(strategy_name, train_per_horizon, df):
    """Kullanılacak doğrulama metodunu (Sabit Baseline veya Walk-Forward) hazırlar."""
    all_strategy_defs = {
        "baseline": (train_per_horizon, None),
        "walkfwd_expanding": (make_dynamic_wf_windows(df, "2023-01-01", 6), None),
    }

    if strategy_name not in all_strategy_defs:
        print(f"  [WARNING] Bilinmeyen strateji '{strategy_name}', 'baseline' varsayılan olarak seçildi.")
        strategy_name = "baseline"

    train_source, strategy_sw = all_strategy_defs[strategy_name]
    return [(strategy_name, train_source, strategy_sw)], strategy_name


def build_model(model_type, seq_len, n_feat):
    """Tez metni Tablo 3.2'deki katman yapısına (Dense, GRU, LSTM) göre yapay sinir ağını inşa eder."""
    from tensorflow.keras.layers import Flatten, GRU, LSTM
    
    if model_type == "Dense":  # İleri Beslemeli Çok Katmanlı Algılayıcı (MLP)
        model = Sequential([
            Input(shape=(seq_len, n_feat)),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(8, activation="relu"),
            Dense(1),  # Çıkış katmanı (Doğrusal / Linear Aktivasyon)
        ])
    elif model_type == "GRU":  # Kapılı Yinelemeli Birim (Tezinizde kullanılan nihai model)
        model = Sequential([
            Input(shape=(seq_len, n_feat)),
            GRU(64, return_sequences=False),  # 64 Hücreli GRU katmanı
            Dense(8, activation="relu"),      # 8 Nöronlu ReLU katmanı
            Dense(1),                         # Tahmin Çıktısı (Doğrusal)
        ])
    elif model_type == "LSTM":  # Uzun Kısa Vadeli Bellek Modeli
        model = Sequential([
            Input(shape=(seq_len, n_feat)),
            LSTM(64, return_sequences=False),
            Dense(8, activation="relu"),
            Dense(1),
        ])
    else:
        raise ValueError(f"Bilinmeyen model_type '{model_type}'. Şunlardan birini seçin: Dense, GRU, LSTM.")

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])  # Kayıp fonksiyonu MSE, Optimizasyon Adam
    return model


def build_and_train(model_type, X_tr_sc, y_tr_sc, X_te_sc, seq_len=1, sample_weight=None):
    """Derin öğrenme modelini derler, Örneklem Ağırlıklandırması ve Erken Durdurma (EarlyStopping) ile eğitir."""
    set_random_seed(RANDOM_SEED)
    n_feat = X_tr_sc.shape[-1]
    model = build_model(model_type, seq_len, n_feat)
    
    # Tezde geçen sabır değeri (patience=8) olan Erken Durdurma mekanizması
    early_stopping = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

    if sample_weight is not None:  # Örneklem ağırlıklandırma aktifse devreye giren eğitim mekanizması
        val_cut = int(len(X_tr_sc) * 0.8)
        Xtr_t, Xtr_v = X_tr_sc[:val_cut], X_tr_sc[val_cut:]
        ytr_t, ytr_v = y_tr_sc[:val_cut], y_tr_sc[val_cut:]
        sw_t = sample_weight[:val_cut]
        history = model.fit(
            Xtr_t,
            ytr_t,
            sample_weight=sw_t,
            validation_data=(Xtr_v, ytr_v),
            epochs=200,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0,
        )
    else:  # Standart eğitim mekanizması
        history = model.fit(
            X_tr_sc,
            y_tr_sc,
            validation_split=0.2,
            shuffle=False,
            epochs=200,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0,
        )

    print(f"      Eğitim tamamlanan epoch sayısı: {len(history.history['loss'])}", end="  ")
    return model.predict(X_te_sc, verbose=0)


def make_sequences(X, seq_len):
    """Girdileri derin ağın (GRU/LSTM) okuyabileceği 3 boyutlu [Örnek Sayısı, Zaman Adımı, Öznitelik] dizilerine dönüştürür."""
    if seq_len <= 1:
        return X[:, np.newaxis, :]

    out = []
    for i in range(seq_len - 1, len(X)):
        out.append(X[i - seq_len + 1: i + 1])

    return np.array(out)


def get_lag_columns(horizon_config):
    """Konfigürasyondan gerekli olan gecikme (lag) kolon isimlerini listeler."""
    return [f"lag_{l}" for l in sorted(set(l for hh in horizon_config.values() for l in hh["lags"]))]


def make_horizon_inputs(data, drop_h, target_col_h):
    """Giriş matrisi (X) ile hedef değişken vektörünü (y) birbirinden ayırır."""
    X = data.drop(columns=[c for c in drop_h if c in data.columns])
    y = data[[target_col_h]]
    return X, y


def evaluate_walk_forward_horizon(train_source, drop_h, target_col_h, model_type, seq_len, halflives):
    """Walk-Forward doğrulama yöntemiyle modeli her 6 aylık adımda eğitip tahmin üreten fonksiyon."""
    all_y_true_h = []
    all_y_pred_h = []

    for train_slice, test_slice in train_source:
        X_tr_h, y_tr_h = make_horizon_inputs(train_slice, drop_h, target_col_h)
        X_te_h, y_te_h = make_horizon_inputs(test_slice, drop_h, target_col_h)

        # MinMax Ölçekleme (Veriyi 0 ile 1 arasına sıkıştırır)
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_tr_sc = scaler_x.fit_transform(X_tr_h)
        y_tr_sc = scaler_y.fit_transform(y_tr_h.values)

        # Zaman serisi pencerelerinin oluşturulması
        X_tr_seq = make_sequences(X_tr_sc, seq_len)
        y_tr_seq = y_tr_sc[seq_len - 1:]
        sample_weights = make_sample_weights(train_slice.index[seq_len - 1:], halflife_days=halflives[0])

        # Test setinin başlangıcındaki geçmiş bağlam (context) kaybını önlemek için birleştirme işlemi
        context_h = X_tr_h.iloc[-(seq_len - 1):] if seq_len > 1 else X_tr_h.iloc[0:0]
        X_te_with_ctx = pd.concat([context_h, X_te_h])
        X_te_sc_ctx = scaler_x.transform(X_te_with_ctx)
        X_te_seq = make_sequences(X_te_sc_ctx, seq_len)

        # Modelin eğitilmesi ve tahminin ters ölçeklenmesi (Inverse Transform)
        pred_sc = build_and_train(model_type, X_tr_seq, y_tr_seq, X_te_seq, seq_len=seq_len, sample_weight=sample_weights)
        pred_us = scaler_y.inverse_transform(pred_sc)

        all_y_true_h.append(y_te_h.values)
        all_y_pred_h.append(pred_us)

    return np.vstack(all_y_true_h).ravel(), np.vstack(all_y_pred_h).ravel(), 0


def evaluate_baseline_horizon(train_source, test_data, y_test, drop_h, target_col_h, model_type, seq_len, strategy_sw):
    """Sabit veri bölünmesine (Baseline) dayalı eğitim ve test tahmin sürecini yönetir."""
    train_data_b = train_source

    if len(train_data_b) < seq_len + 10:
        print(f"    [WARNING] Satır sayısı seq_len={seq_len} için çok az, atlanıyor.")
        return None

    X_tr_h, y_tr_h = make_horizon_inputs(train_data_b, drop_h, target_col_h)
    X_test_h = test_data.drop(columns=[c for c in drop_h if c in test_data.columns])

    sc_X = MinMaxScaler()
    sc_y = MinMaxScaler()
    X_tr_sc = sc_X.fit_transform(X_tr_h)
    y_tr_sc = sc_y.fit_transform(y_tr_h.values)
    X_te_sc = sc_X.transform(X_test_h)

    X_tr_seq = make_sequences(X_tr_sc, seq_len)
    X_te_seq = make_sequences(X_te_sc, seq_len)
    y_tr_seq = y_tr_sc[seq_len - 1:]

    pred_sc = build_and_train(model_type, X_tr_seq, y_tr_seq, X_te_seq, seq_len=seq_len, sample_weight=strategy_sw)
    y_pred_h = sc_y.inverse_transform(pred_sc).ravel()
    y_true_h = y_test[target_col_h].values[seq_len - 1:]

    return y_true_h, y_pred_h, len(train_data_b)


def calculate_metrics(y_true_h, y_pred_h):
    """Tez Bölüm 3.5.2'deki 'Fiziksel Sınır Kısıtı (clip)' işlemini yapar ve başarı metriklerini (RMSE, MAE, R²) hesaplar."""
    y_pred_h = np.clip(y_pred_h, 0, 100)  # Tahminleri %0 ile %100 arasına sabitler (Fiziksel sınır kısıtı)
    rmse_h = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
    mae_h = mean_absolute_error(y_true_h, y_pred_h)
    r2_h = r2_score(y_true_h, y_pred_h)
    return y_pred_h, rmse_h, mae_h, r2_h


def plot_predictions(dam, model_type, strategy_label, horizon_metrics, horizon_config, y_test, is_walkfwd):
    """Tez Bölüm 4 (Bulgular) için gerçek vs. tahmin grafiklerini ve kalıntı (residual) grafiklerini çizip kaydeder."""
    fig, axes = plt.subplots(6, 1, figsize=(14, 22), gridspec_kw={"height_ratios": [3, 1, 3, 1, 3, 1]})
    fig.suptitle(
        f"{dam} Dam [{model_type} | {strategy_label}] — Multi-Horizon Fill % Prediction vs Actual",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    horizon_colors = ["#1f77b4", "#2ca02c", "#9467bd"]
    pred_colors = ["#ff7f0e", "#d62728", "#8c564b"]

    for i, (h, hcolor, pcolor) in enumerate(zip(TARGET_HORIZONS, horizon_colors, pred_colors)):
        label = f"+{h}d"
        hm = horizon_metrics[label]
        y_true_h = hm["y_true"]
        y_pred_h = hm["y_pred"]
        seq_len = horizon_config[h]["seq_len"]
        trim = 0 if is_walkfwd else seq_len - 1
        pred_index = y_test.index[trim: trim + len(y_true_h)] + pd.Timedelta(days=h)

        ax = axes[i * 2]
        ax2 = axes[i * 2 + 1]

        # Gerçek ve tahmin edilen değerlerin çizdirilmesi
        ax.plot(pred_index, y_true_h, label="Actual", color=hcolor, linewidth=1.8)
        ax.plot(pred_index, y_pred_h, label=f"Predicted {label}", color=pcolor, linewidth=1.8, linestyle="--")
        ax.fill_between(pred_index, y_true_h, y_pred_h, alpha=0.15, color="gray", label="Error band")
        ax.set_ylabel("Fill Level (%)", fontsize=10)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Horizon {label}", fontsize=11, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")

        # Grafik üzerine RMSE, MAE ve R² metriklerinin kutucuk olarak eklenmesi
        metrics_text = f"RMSE: {hm['rmse']:.2f}%   MAE: {hm['mae']:.2f}%   R²: {hm['r2']:.4f}"
        ax.text(
            0.01,
            0.04,
            metrics_text,
            transform=ax.transAxes,
            fontsize=9,
            color="dimgray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

        # Hata paylarını gösteren Kalıntı (Residual) bar grafiklerinin çizdirilmesi
        residuals = y_true_h - y_pred_h
        ax2.bar(pred_index, residuals, color=["#d62728" if r < 0 else "#2ca02c" for r in residuals], width=1.0, alpha=0.7)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_ylabel("Residual (%)", fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=35, ha="right")

        if i == len(TARGET_HORIZONS) - 1:
            ax2.set_xlabel("Date", fontsize=11)

    plt.tight_layout()
    out_path = f"{OUT_DIR}/{dam}_{strategy_label}_{model_type}_prediction.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")  # Grafiği sonuç klasörüne kaydeder
    plt.close()
    print(f"    Grafik kaydedildi: {out_path}")


def summarize_run(horizon_metrics, train_rows, test_rows):
    """Hesaplanan tüm metrik sonuçlarını bir sözlük (dictionary) yapısında özetler."""
    expected_labels = {f"+{h}d" for h in TARGET_HORIZONS}
    missing_labels = expected_labels - set(horizon_metrics.keys())

    if missing_labels:
        warn_once(
            "missing_horizon_metrics",
            f"Bazı tahmin ufukları için metrik üretilemedi: {sorted(missing_labels)}.",
        )

    return {
        "r2_7d": horizon_metrics["+7d"]["r2"],
        "r2_14d": horizon_metrics["+14d"]["r2"],
        "r2_30d": horizon_metrics["+30d"]["r2"],
        "rmse_7d": horizon_metrics["+7d"]["rmse"],
        "rmse_14d": horizon_metrics["+14d"]["rmse"],
        "rmse_30d": horizon_metrics["+30d"]["rmse"],
        "mae_7d": horizon_metrics["+7d"]["mae"],
        "mae_14d": horizon_metrics["+14d"]["mae"],
        "mae_30d": horizon_metrics["+30d"]["mae"],
        "train_rows": train_rows,
        "test_rows": test_rows,
    }


def run_dam(dam, cfg_override=None, startyear=None, save_plots=True):
    """Tek bir baraj için veri yükleme, eğitim, tahmin ve değerlendirme adımlarını ardışık çalıştıran ana motor."""
    dam_config = get_dam_config(dam, cfg_override)
    prod_cols = dam_config["prod_cols"]
    weather_stations = dam_config["weather_stations"]
    lag_days = dam_config["lag_days"]
    model_type = dam_config["model_type"]
    seq_lens = dam_config["seq_lens"]
    halflives = dam_config["halflives"]
    strategy_name = dam_config["strategy_name"]

    horizon_config = make_horizon_config(lag_days, seq_lens)

    target_cols_multi = [f"target_{h}" for h in TARGET_HORIZONS]
    df, target_col = build_feature_dataframe(dam, prod_cols, weather_stations, lag_days)
    df = limit_dam_date_range(df, dam, target_col, startyear)

    if df is None:
        return None

    pre2023, test_data = split_train_test(df, target_col)

    if len(test_data) == 0:
        print(f"  [WARNING] {dam} için test verisi bulunamadı, atlanıyor.\n")
        return None

    drop_from_X = [target_col] + target_cols_multi
    y_test = test_data[target_cols_multi]
    strategies, strategy_name = make_strategies(strategy_name, pre2023, df)
    lag_cols_all = get_lag_columns(horizon_config)

    for strategy_label, train_source, strategy_sw in strategies:
        is_walkfwd = strategy_label.startswith("walkfwd")
        print(f"\n  [{dam}] strateji={strategy_label}  model={model_type}  pencereler={seq_lens}")
        horizon_metrics = {}

        # Her bir tahmin ufku için döngü halinde modellerin çalıştırılması
        for h in TARGET_HORIZONS:
            h_cfg = horizon_config[h]
            h_lags = h_cfg["lags"]
            seq_len = h_cfg["seq_len"]
            target_col_h = f"target_{h}"
            label = f"+{h}d"
            lag_cols_h = [f"lag_{l}" for l in h_lags]
            lag_cols_drop = [c for c in lag_cols_all if c not in lag_cols_h]
            drop_h = drop_from_X + lag_cols_drop

            # Strateji tipine göre doğru değerlendirme fonksiyonunun çağrılması
            if is_walkfwd:
                result = evaluate_walk_forward_horizon(train_source, drop_h, target_col_h, model_type, seq_len, halflives)
            else:
                train_h = train_source[h]
                result = evaluate_baseline_horizon(train_h, test_data, y_test, drop_h, target_col_h, model_type, seq_len, strategy_sw)

            if result is None:
                continue

            # Performans metriklerinin hesaplanması ve konsola basılması
            y_true_h, y_pred_h, train_rows = result
            y_pred_h, rmse_h, mae_h, r2_h = calculate_metrics(y_true_h, y_pred_h)
            horizon_metrics[label] = {
                "rmse": rmse_h,
                "mae": mae_h,
                "r2": r2_h,
                "y_true": y_true_h,
                "y_pred": y_pred_h,
            }
            print(f"    [{label}]  RMSE: {rmse_h:.2f}%  MAE: {mae_h:.2f}%  R²: {r2_h:.4f}")

        if save_plots:
            plot_predictions(dam, model_type, strategy_label, horizon_metrics, horizon_config, y_test, is_walkfwd)

        return summarize_run(horizon_metrics, train_rows, len(test_data))


def print_dam_config(dam, cfg):
    """Model çalışmadan önce hangi barajın hangi hiperparametrelerle eğitime gireceğini konsola özetler."""
    prod_cols = cfg.get("prod_cols", [])
    weather_stations = cfg.get("weather_stations", [])
    lag_days = cfg.get("lag_days", DEFAULT_LAG_DAYS)
    model_type = cfg.get("model", "GRU")
    seq_lens = cfg.get("seq_lens", [30, 60, 90])
    halflives = cfg.get("halflives", [365])
    strategy_name = cfg.get("strategy", "baseline")

    print(f"  Üretim/Talep Kolonları ({len(prod_cols)}): {prod_cols}")
    print(f"  Hava İstasyonları     ({len(weather_stations)}): {weather_stations}")
    print(f"  Gecikme Günleri       : {lag_days}")
    print(f"  Model Mimarisi        : {model_type}")
    # ... (Diğer parametre bilgileri ekrana basılır)


def make_summary_row(dam, strategy_name, model_type, result):
    """Nihai DataFrame özeti için satır verisi oluşturur."""
    return {
        "Dam": dam,
        "Strategy": strategy_name,
        "Model": model_type,
        "RMSE +7d (%)": round(result["rmse_7d"], 2),
        "MAE +7d (%)": round(result["mae_7d"], 2),
        "R2 +7d": round(result["r2_7d"], 4),
        "RMSE +14d (%)": round(result["rmse_14d"], 2),
        "MAE +14d (%)": round(result["mae_14d"], 2),
        "R2 +14d": round(result["r2_14d"], 4),
        "RMSE +30d (%)": round(result["rmse_30d"], 2),
        "MAE +30d (%)": round(result["mae_30d"], 2),
        "R2 +30d": round(result["r2_30d"], 4),
        "Train rows": result["train_rows"],
        "Test rows": result["test_rows"],
    }


def main():
    """Tüm barajlar üzerinde sırayla dönerek ana eğitim algoritmasını tetikleyen yürütücü fonksiyon."""
    results_summary = []

    for dam in DAMS:
        target_col = f"{dam}DolulukOrani"
        print(f"{'=' * 50}")
        print(f"[{dam}] Model Eğitimi Başlıyor -> Hedef Değişken: {target_col}")
        print(f"{'=' * 50}")

        cfg = FEATURE_CONFIG[dam]
        model_type = cfg.get("model", "GRU")
        strategy_name = cfg.get("strategy", "baseline")
        print_dam_config(dam, cfg)

        # İlgili baraj için modeli koşturur
        result = run_dam(dam, startyear=args.startyear, save_plots=True)

        if result is None:
            continue

        results_summary.append(make_summary_row(dam, strategy_name, model_type, result))

    # Tüm barajların sonuçlarının toplu bir tablo halinde konsola yazdırılması ve CSV olarak kaydedilmesi
    print("\n" + "=" * 60)
    print("ÖZET RAPOR - TÜM BARAJLARI PERFORMANSI")
    print("=" * 60)

    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

    summary_csv = f"{OUT_DIR}/dam_results_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nTüm özet sonuçlar şuraya kaydedildi: {summary_csv}")


if __name__ == "__main__":
    # Programın ilk tetiklendiği ana giriş noktası
    args = parse_arguments()
    DAMS = resolve_dams(args.dams)
    FEATURE_CONFIG = load_feature_config(FEATURE_CONFIG_PATH)
    df_raw = clean_raw_data(load_raw_data())
    os.makedirs(OUT_DIR, exist_ok=True)  # Çıktı klasörü yoksa otomatik oluşturur
    set_random_seed()
    main()
