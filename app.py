import gradio as gr
import numpy
import librosa
import joblib
import config

# --- 1. TẢI MODEL VÀ SCALER ---
# Tải các đối tượng đã được huấn luyện và lưu lại.
# Việc này chỉ thực hiện một lần khi ứng dụng khởi động.
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and Scaler loaded successfully.")
except FileNotFoundError:
    print("ERROR: model.pkl or scaler.pkl not found. Please run the training script first.")
    # Thoát ứng dụng nếu không tìm thấy model hoặc scaler
    exit()

# --- 2. CÁC THAM SỐ VÀ HÀM CẦN THIẾT ---
# Lấy các tham số từ file config.py
SAMPLING_RATE = config.CreateDataset.SAMPLING_RATE #
FRAME_SIZE = config.CreateDataset.FRAME_SIZE #
HOP_SIZE = config.CreateDataset.HOP_SIZE #

# Danh sách các thể loại theo đúng thứ tự đã được huấn luyện
# Thường là thứ tự alphabet của các thư mục
GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

# Sao chép y hệt hàm trích xuất đặc trưng từ file training/recognition
def extract_features(signal, sample_rate, frame_size, hop_size): #
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_size, hop_length=hop_size) #
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size) #
    spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size) #
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size) #
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size) #
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size) #

    # Tính toán giá trị trung bình và độ lệch chuẩn của các đặc trưng
    return [
        numpy.mean(zero_crossing_rate), numpy.std(zero_crossing_rate), #
        numpy.mean(spectral_centroid), numpy.std(spectral_centroid), #
        numpy.mean(spectral_contrast), numpy.std(spectral_contrast), #
        numpy.mean(spectral_bandwidth), numpy.std(spectral_bandwidth), #
        numpy.mean(spectral_rolloff), numpy.std(spectral_rolloff), #
        numpy.mean(mfccs[1, :]), numpy.std(mfccs[1, :]), #
        numpy.mean(mfccs[2, :]), numpy.std(mfccs[2, :]), #
        numpy.mean(mfccs[3, :]), numpy.std(mfccs[3, :]), #
        numpy.mean(mfccs[4, :]), numpy.std(mfccs[4, :]), #
        numpy.mean(mfccs[5, :]), numpy.std(mfccs[5, :]), #
        numpy.mean(mfccs[6, :]), numpy.std(mfccs[6, :]), #
        numpy.mean(mfccs[7, :]), numpy.std(mfccs[7, :]), #
        numpy.mean(mfccs[8, :]), numpy.std(mfccs[8, :]), #
        numpy.mean(mfccs[9, :]), numpy.std(mfccs[9, :]), #
        numpy.mean(mfccs[10, :]), numpy.std(mfccs[10, :]), #
        numpy.mean(mfccs[11, :]), numpy.std(mfccs[11, :]), #
        numpy.mean(mfccs[12, :]), numpy.std(mfccs[12, :]), #
        numpy.mean(mfccs[13, :]), numpy.std(mfccs[13, :]), #
    ]


# --- 3. HÀM DỰ ĐOÁN CHÍNH ---
# Hàm này sẽ được Gradio gọi mỗi khi người dùng nhấn nút "Submit"
def predict_genre(audio):
    # Gradio truyền vào một tuple: (sample_rate, numpy_array)
    # Kiểm tra nếu người dùng không upload file nào
    if audio is None:
        return "Please upload an audio file."

    sample_rate_in, signal_in = audio
    
    # === THÊM DÒNG NÀY ĐỂ SỬA LỖI ===
    # Chuyển đổi tín hiệu âm thanh từ số nguyên sang số thực
    signal_in = signal_in.astype(numpy.float32)
    # ================================

    # Nếu tần số lấy mẫu của file input khác với lúc train, cần resample
    if sample_rate_in != SAMPLING_RATE:
        signal_in = librosa.resample(y=signal_in, orig_sr=sample_rate_in, target_sr=SAMPLING_RATE)

    # Trích xuất đặc trưng từ tín hiệu âm thanh
    features = extract_features(signal_in, SAMPLING_RATE, FRAME_SIZE, HOP_SIZE)
    
    # Chuyển thành numpy array và reshape để có dạng (1, n_features)
    features_reshaped = numpy.array(features).reshape(1, -1)

    # --- SỬ DỤNG SCALER ĐÃ LƯU ---
    # Dùng .transform() chứ KHÔNG phải .fit_transform()
    features_scaled = scaler.transform(features_reshaped)

    # --- DỰ ĐOÁN ---
    # prediction_index = model.predict(features_scaled)
    
    # # Lấy tên thể loại từ danh sách
    # predicted_genre = prediction_index[0]

    predicted_genre = model.predict(features_scaled)[0]

    return predicted_genre

# --- 4. XÂY DỰNG VÀ KHỞI CHẠY GIAO DIỆN GRADIO ---
iface = gr.Interface(
    fn=predict_genre,
    inputs=gr.Audio(type="numpy", label="Tải lên file âm thanh của bạn"),
    outputs=gr.Label(label="Thể loại dự đoán"),
    title="Music Genre Classification",
    description="Tải lên một file âm thanh (.wav, .mp3) để AI dự đoán thể loại nhạc. Model được huấn luyện trên bộ dữ liệu GTZAN."
)

if __name__ == "__main__":
    iface.launch(share=True)