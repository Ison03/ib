
from datetime import datetime, timedelta
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String
from sqlalchemy.orm import declarative_base, Session
import paho.mqtt.client as mqtt
import json
import numpy as np
import pandas as pd
import altair as alt
import random
import threading
import time as tm
import streamlit as st
import smtplib
from email.mime.text import MIMEText

ADMIN_EMAIL = "admin@example.com"    # сюда «шлём алёрты»
FROM_EMAIL = "monitor@example.com"   # технический адрес (можно заглушку)

def notify_admin(subject: str, body: str):
    """Отправка письма администратору (прототип)."""
    msg = MIMEText(body, _charset="utf-8")
    msg["Subject"] = subject
    msg["From"] = FROM_EMAIL
    msg["To"] = ADMIN_EMAIL

    try:
        # это пример для локального SMTP, в реале будут настройки сервера
        with smtplib.SMTP("smtp.example.com", 587) as server:
            server.starttls()
            server.login("user", "password")  # в реале — из настроек/переменных окружения
            server.send_message(msg)
    except Exception as e:
        print("Не удалось отправить уведомление:", e)

HOSTS = ["PC-ACCOUNTING", "PC-CEO", "PC-DEV-01", "PC-DEV-02", "PC-ADMIN"]


# ==================== НАСТРОЙКА СТРАНИЦЫ ====================
st.set_page_config("⚡ SmartMonitor AI", layout="wide")

# ==================== НАСТРОЙКА БАЗЫ ДАННЫХ ====================
DB_URL = "sqlite:///smartmonitor.db"
engine = create_engine(DB_URL, echo=False)
Base = declarative_base()

MQTT_BROKER = "test.mosquitto.org"  # или IP локального брокера
MQTT_PORT = 1883
MQTT_TOPIC = "smartmonitor/data"

def on_message(client, userdata, message):
    try:
        payload = json.loads(message.payload.decode())
        voltage = float(payload.get("voltage", 0))
        load = float(payload.get("network_load", 0))

        new_row = pd.DataFrame([{
            "time": datetime.now().strftime("%H:%M:%S"),
            "voltage": voltage,
            "network_load": load
        }])
        new_data = detect_anomalies(new_row)
        save_readings(new_data)
        st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True).tail(400)
    except Exception as e:
        print("Ошибка обработки MQTT:", e)





def save_readings(df):
    with Session(engine) as session:
        for _, row in df.iterrows():
            time = row["time"]
            # на всякий случай приводим time к строке
            if not isinstance(time, str):
                time = str(time)

            session.add(Reading(
                time=time,
                host=row["host"],   # ← сохраняем компьютер
                voltage=row["voltage"],
                network_load=row["network_load"],
                recon_error=row.get("recon_error", 0.0),
                anomaly=int(row.get("anomaly", 0)),
            ))
        session.commit()

def load_history(limit=500):
    """Загружаем последние N записей"""
    with Session(engine) as session:
        rows = session.query(Reading).order_by(Reading.id.desc()).limit(limit).all()
        data = [
            (r.time, r.voltage, r.network_load, r.recon_error, r.anomaly)
            for r in reversed(rows)
        ]
        return pd.DataFrame(data, columns=["time", "voltage", "network_load", "recon_error", "anomaly"])

def clear_db():
    """Очищаем таблицу"""
    with Session(engine) as session:
        session.query(Reading).delete()
        session.commit()

class Reading(Base):
    __tablename__ = "readings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(String)
    host = Column(String)
    voltage = Column(Float)
    network_load = Column(Float)
    recon_error = Column(Float)
    anomaly = Column(Integer)

class ModelLog(Base):
    __tablename__ = "model_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.now)
    event = Column(String, default="retrain")  # retrain / auto_retrain
    mean_error = Column(Float)
    n_records = Column(Integer)

Base.metadata.create_all(engine)
# ==================== ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ ====================
if "scaler" not in st.session_state:
    st.session_state.scaler = StandardScaler()

if "model" not in st.session_state:
    input_layer = Input(shape=(2,))
    encoded = Dense(8, activation='relu')(input_layer)
    encoded = Dense(4, activation='relu')(encoded)
    decoded = Dense(8, activation='relu')(encoded)
    output = Dense(2, activation='linear')(decoded)
    autoencoder = Model(inputs=input_layer, outputs=output)
    autoencoder.compile(optimizer=Adam(0.001), loss='mse')
    st.session_state.model = autoencoder
    st.session_state.initial_trained = False
    if "last_retrain" not in st.session_state:
        st.session_state.last_retrain = tm.time()
    if "auto_train_active" not in st.session_state:
        st.session_state.auto_train_active = True

def log_model_update(mean_error, n_records, event="retrain"):
    """Сохраняем запись об обучении/переобучении модели"""
    with Session(engine) as session:
        log = ModelLog(
            timestamp=datetime.now(),
            event=event,
            mean_error=float(mean_error),
            n_records=int(n_records)
        )
        session.add(log)
        session.commit()


if "logged_initial" not in st.session_state:
    log_model_update(mean_error=0.0, n_records=0, event="initial_train")
    st.session_state.logged_initial = True

# ==================== ГЕНЕРАЦИЯ И АНАЛИЗ ====================



# --- ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ ---
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(
        columns=["time", "host", "voltage", "network_load", "recon_error", "anomaly"]
    )

# --- ОДИН ШАГ СИМУЛЯЦИИ (БЕЗ ЦИКЛА!) ---
def generate_data(n=5):
    base = datetime.now()
    rows = []

    for i in range(n):
        ts = base + timedelta(seconds=i * 0.5)  # каждые полсекунды, например

        host = random.choice(HOSTS)   # ← выбираем случайный ПК

        # базовые "нормальные" значения
        voltage = np.random.normal(230, 2)
        load = np.random.normal(10, 3)

        # редкие аномальные всплески (3–5% случаев)
        if np.random.rand() < 0.01:  # 4% – можно менять
            # иногда сразу сильно бьёт и по сети, и по напряжению
            voltage += np.random.choice([+30, +40, -35, -45])
            load    += np.random.choice([+20, +30, +40])

        rows.append({
            "time": ts.isoformat(),   # строка для БД
            "host": host,  # ← НОВОЕ ПОЛЕ
            "voltage": voltage,
            "network_load": load,
        })

    return pd.DataFrame(rows)




if "base_threshold" not in st.session_state:
    st.session_state.base_threshold = None
def detect_anomalies(df):
    """Аномалия = высокая ошибка автоэнкодера + заметный скачок по напряжению или трафику"""

    # === Первичное обучение один раз ===
    if not st.session_state.initial_trained:
        norm = generate_data(1000)
        # фильтр "примерно нормальных" значений
        norm = norm[
            (norm["voltage"].between(220, 236)) &
            (norm["network_load"].between(3, 20))
        ]
        st.session_state.scaler.fit(norm[["voltage", "network_load"]])
        X_train = st.session_state.scaler.transform(norm[["voltage", "network_load"]])
        st.session_state.model.fit(X_train, X_train, epochs=10, verbose=0)
        st.session_state.initial_trained = True

        # калибруем базовый порог по ошибке
        recon_train = st.session_state.model.predict(X_train, verbose=0)
        err_train = np.mean(np.square(X_train - recon_train), axis=1)
        # очень жёсткий порог: только самые редкие значения
        st.session_state.base_threshold = np.median(err_train) + 4.0 * np.std(err_train)

    # === Анализ новых данных ===
    X = st.session_state.scaler.transform(df[["voltage", "network_load"]])
    recon = st.session_state.model.predict(X, verbose=0)
    mse = np.mean(np.square(X - recon), axis=1)
    df["recon_error"] = mse

    # 1) критерий по автоэнкодеру
    ae_flag = mse > st.session_state.base_threshold

    # 2) "физический" критерий по самих величинам (грубый, но понятный)
    #   здесь можно подстроить цифры под твой генератор / реальные данные
    volt_jump = np.abs(df["voltage"] - df["voltage"].mean())
    load_jump = np.abs(df["network_load"] - df["network_load"].mean())

    volt_flag = volt_jump > 8   # например, отклонение по напряжению > 8 В
    load_flag = load_jump > 10  # отклонение по трафику > 10 Мбит/с

    physical_flag = volt_flag | load_flag

    # Итоговая аномалия = и нейросеть, и физический порог согласны
    df["anomaly"] = ((ae_flag) & (physical_flag)).astype(int)

    return df


def get_model_confidence(data):
    """Анализ состояния модели на основе последних данных"""
    if data.empty:
        return "⚪", "Недостаточно данных"

    last = data.tail(100)  # последние 100 измерений
    mean_error = float(last["recon_error"].mean())
    anomaly_rate = float((last["anomaly"] == 1).sum() / len(last))

    # Гибкий анализ
    if anomaly_rate < 0.02 and mean_error < 0.02:
        return "🟢", "Стабильна — сеть уверена в данных"
    elif anomaly_rate < 0.08 or mean_error < 0.05:
        return "🟡", "Адаптируется — анализирует новые условия"
    else:
        return "🔴", "Перегрузка — требуется переобучение"



def auto_self_train():
    """Фоновое самообучение модели — адаптация и автопереобучение при 'перегрузке'"""
    while st.session_state.auto_train_active:
        tm.sleep(600)  # каждые 10 минут (можно изменить)

        history = load_history(1000)
        if history.empty:
            continue

        # Определяем текущее состояние модели
        emoji, state_text = get_model_confidence(history)
        print(f"[AUTO TRAIN] {emoji} {state_text} @ {datetime.now().strftime('%H:%M:%S')}")

        # === Реакция на состояние ===
        if emoji == "🟢":
            # Всё стабильно, можно чуть "успокоить" порог
            st.session_state.base_threshold *= 1.02

        elif emoji == "🟡":
            # Модель адаптируется — немного дообучаем
            normals = history[history["anomaly"] == 0]
            if len(normals) > 100:
                X = st.session_state.scaler.fit_transform(normals[["voltage","network_load"]])
                st.session_state.model.fit(X, X, epochs=2, verbose=0)
            st.session_state.base_threshold *= 0.98  # чуть чувствительнее

        elif emoji == "🔴":
            # Система перегружена — выполняем переобучение
            normals = history[history["anomaly"] == 0]
            if len(normals) > 200:
                X = st.session_state.scaler.fit_transform(normals[["voltage","network_load"]])
                st.session_state.model.fit(X, X, epochs=8, verbose=0)
                log_model_update(
                    mean_error=float(np.mean(np.square(X - st.session_state.model.predict(X, verbose=0)))),
                    n_records=len(normals),
                    event="auto_retrain"
                )
                st.session_state.base_threshold *= 1.05  # сделаем менее нервной
                print(f"[AUTO TRAIN] 🔁 Модель автоматически переобучена ({len(normals)} данных)")

threading.Thread(target=auto_self_train, daemon=True).start()

if "auto_train_active" not in st.session_state:
    threading.Thread(target=auto_self_train, daemon=True).start()

def start_mqtt_listener():
    """Фоновый поток для приёма данных от датчиков"""
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT)
    client.subscribe(MQTT_TOPIC)
    client.loop_forever()

# Запуск слушателя в отдельном потоке
threading.Thread(target=start_mqtt_listener, daemon=True).start()
def load_model_logs(limit=50):
    """Загружаем историю обучения модели из базы данных"""
    from sqlalchemy import desc
    with Session(engine) as session:
        logs = session.query(ModelLog).order_by(desc(ModelLog.id)).limit(limit).all()
        if not logs:
            return pd.DataFrame(columns=["time", "event", "mean_error", "n_records"])
        data = [
            (l.timestamp, l.event, l.mean_error, l.n_records)
            for l in reversed(logs)
        ]
        return pd.DataFrame(data, columns=["time", "event", "mean_error", "n_records"])


def simulate_mqtt_data():
    import paho.mqtt.publish as publish
    while True:
        data = {
            "voltage": np.random.normal(228, 2),
            "network_load": np.random.normal(10, 3)
        }
        publish.single(MQTT_TOPIC, json.dumps(data), hostname=MQTT_BROKER, port=MQTT_PORT)
        tm.sleep(1)

# ==================== UI ====================
st.sidebar.title("⚙️ Управление системой")

reset_button = st.sidebar.button("🔁 Сбросить модель")
clear_db_button = st.sidebar.button("🗑️ Очистить базу данных")

st.sidebar.subheader("Фильтр по компьютеру")

df_full = st.session_state.data

# защита от ситуации, когда данных ещё нет или нет поля host
if not df_full.empty and "host" in df_full.columns:
    hosts = sorted(df_full["host"].dropna().unique().tolist())
else:
    hosts = []

selected_host = st.sidebar.selectbox(
    "Компьютер",
    options=["Все"] + hosts
)

if selected_host != "Все":
    df_view = st.session_state.data[st.session_state.data["host"] == selected_host]
else:
    df_view = st.session_state.data


if clear_db_button:
    clear_db()
    st.sidebar.warning("База очищена!")

placeholder = st.empty()

# Загружаем историю
history = load_history(300)
st.session_state.data = history if not history.empty else pd.DataFrame(columns=["time","voltage","network_load","recon_error","anomaly"])

# Основной цикл
# === Основной интерфейс ===

st.title("💡 SmartMonitor — интеллектуальная система анализа аномалий")

# === Вкладки интерфейса ===
tab1, tab2 = st.tabs(["📊 Мониторинг", "🧭 Состояние системы"])

# ==================== 📊 TAB 1: Мониторинг ====================
with tab1:

    placeholder = st.empty()
    for _ in range(5000):  # количество обновлений (можно увеличить)
        new_data = generate_data(5)
        new_data = detect_anomalies(new_data)

        st.session_state.data = pd.concat(
            [st.session_state.data, new_data]
        ).tail(1000)
        save_readings(new_data)



        with placeholder.container():
            col1, col2 = st.columns(2)

            # === Графики ===
            with col1:
                st.subheader("📈 Напряжение и сетевой трафик")

                if not st.session_state.data.empty:
                    df_plot = st.session_state.data.tail(300)

                    # если time уже datetime в generate_data, можно даже не конвертировать
                    df_plot["time"] = pd.to_datetime(df_plot["time"], errors="coerce")
                    df_plot = df_plot.set_index("time")

                    st.line_chart(df_plot[["voltage", "network_load"]])
                    st.caption(f"Точек в окне: {len(df_plot)}")
                else:
                    st.info("Данные пока не получены.")
                anom_last = st.session_state.data[st.session_state.data["anomaly"] == 1].tail(10)
                if not anom_last.empty:
                    st.dataframe(anom_last[["time", "host", "voltage", "network_load", "recon_error"]],
                                 hide_index=True, use_container_width=True)
                else:
                    st.info("Пока аномалий не было.")

            # === Метрики и аномалии ===
            with col2:
                st.subheader("🚨 Обнаруженные аномалии")
                anomalies_count = int((st.session_state.data["anomaly"] == 1).sum())
                total_count = len(st.session_state.data)
                percent = (anomalies_count / total_count * 100) if total_count else 0
                st.metric("Обнаружено аномалий", f"{anomalies_count} ({percent:.1f}%)")

                # Таблица последних данных
                st.dataframe(
                    st.session_state.data.tail(1000),
                    hide_index=True,
                    use_container_width=True
                )

                # === Индикатор уверенности модели ===
                emoji, state_text = get_model_confidence(st.session_state.data)
                st.markdown(f"### {emoji} Состояние модели: {state_text}")

                # Цветовые уведомления
                if emoji == "🔴":
                    st.warning("⚠️ Модель перегружена. Выполняется автоматическое переобучение…")
                elif emoji == "🟡":
                    st.info("ℹ️ Модель адаптируется к новым данным.")
                elif emoji == "🟢":
                    st.success("✅ Модель работает стабильно.")
                else:
                    st.info("🤖 Модель ожидает данных.")
                st.subheader("Распределение аномалий по компьютерам")

                if not st.session_state.data.empty:
                    anom = st.session_state.data[st.session_state.data["anomaly"] == 1]

                    if not anom.empty:
                        counts = (
                            anom.groupby("host")["anomaly"]
                            .count()
                            .reset_index()
                            .rename(columns={"anomaly": "anomaly_count"})
                        )

                        st.bar_chart(counts.set_index("host"))
                        st.dataframe(counts, hide_index=True, use_container_width=True)

                    else:
                        st.info("Аномалий пока не обнаружено.")
                else:
                    st.info("Данных ещё нет.")

        tm.sleep(0.1)  # задержка обновления графика (1 секунда)

# ==================== 🧭 TAB 2: Состояние системы ====================
with tab2:
    st.subheader("🧠 Текущее состояние SmartMonitor")

    logs = load_model_logs()
    auto_retrains = len(logs[logs["event"] == "auto_retrain"]) if not logs.empty else 0

    mean_error = (
        float(st.session_state.data["recon_error"].mean())
        if not st.session_state.data.empty else 0
    )
    network_temp = (
        float(st.session_state.data["network_load"].tail(50).mean())
        if not st.session_state.data.empty else 0
    )

    emoji, state_text = get_model_confidence(st.session_state.data)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🤖 Уверенность модели", emoji)
    with col2:
        st.metric("🔁 Автопереобучений", auto_retrains)
    with col3:
        st.metric("📉 Средняя ошибка", f"{mean_error:.3f}")
    with col4:
        st.metric("🌡️ Температура сети", f"{network_temp:.1f} Мбит/с")

    st.divider()
    st.subheader("📊 История обучения модели")

    if not logs.empty:
        st.line_chart(logs.set_index("time")[["mean_error"]])
        st.dataframe(logs.tail(10), hide_index=True, use_container_width=True)
    else:
        st.info("История обучения пока пуста.")
