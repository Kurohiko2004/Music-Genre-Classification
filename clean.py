import os
import shutil
import librosa
import logging

# --- Cấu hình ---
# Thiết lập logging để ghi lại tiến trình và lỗi một cách rõ ràng
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Đường dẫn đến thư mục chứa các thể loại nhạc gốc (có thể có file lỗi)
SOURCE_DIR = os.path.join("TrainFiles", "genres_original")

# Đường dẫn đến thư mục mới để lưu các file đã được xác thực là hợp lệ
CLEANED_DIR = os.path.join("TrainFiles", "genres_cleaned")

def clean_dataset(source_path, cleaned_path):
    """
    Duyệt qua bộ dữ liệu nguồn, kiểm tra từng tệp âm thanh bằng librosa
    và di chuyển các tệp hợp lệ sang một thư mục mới.
    """
    # Nếu thư mục nguồn không tồn tại, dừng lại và báo lỗi
    if not os.path.exists(source_path):
        logging.error(f"Thư mục nguồn không tồn tại: {source_path}")
        return

    # Tạo thư mục đích nếu nó chưa tồn tại
    os.makedirs(cleaned_path, exist_ok=True)
    logging.info(f"Đã tạo hoặc xác nhận thư mục đích: {cleaned_path}")

    # Đếm số file hợp lệ và file lỗi
    valid_files_count = 0
    corrupted_files_count = 0
    
    # Bắt đầu duyệt qua các thư mục con (genres) trong thư mục nguồn
    for genre_folder in os.listdir(source_path):
        source_genre_path = os.path.join(source_path, genre_folder)
        
        # Chỉ xử lý nếu là thư mục
        if os.path.isdir(source_genre_path):
            cleaned_genre_path = os.path.join(cleaned_path, genre_folder)
            os.makedirs(cleaned_genre_path, exist_ok=True) # Tạo thư mục thể loại trong thư mục đích
            
            logging.info(f"--- Đang xử lý thể loại: {genre_folder} ---")
            
            # Duyệt qua từng file trong thư mục thể loại
            for filename in os.listdir(source_genre_path):
                if filename.endswith(".wav"): # Chỉ xử lý file .wav
                    file_path = os.path.join(source_genre_path, filename)
                    
                    try:
                        # Thử tải file âm thanh bằng librosa
                        # Đây là bước kiểm tra quan trọng nhất. Nếu không có lỗi, file hợp lệ.
                        y, sr = librosa.load(file_path, mono=True, duration=5)
                        
                        # Nếu tải thành công, di chuyển file sang thư mục "cleaned"
                        destination_path = os.path.join(cleaned_genre_path, filename)
                        shutil.move(file_path, destination_path)
                        logging.info(f"[HỢP LỆ] Đã di chuyển: {filename}")
                        valid_files_count += 1
                        
                    except Exception as e:
                        # Nếu có lỗi xảy ra khi tải file, báo lỗi và bỏ qua
                        logging.error(f"[LỖI] Không thể xử lý file: {filename}. Lý do: {e}")
                        corrupted_files_count += 1
                        
    logging.info("--- HOÀN TẤT ---")
    logging.info(f"Tổng số file hợp lệ: {valid_files_count}")
    logging.info(f"Tổng số file bị lỗi/bỏ qua: {corrupted_files_count}")


if __name__ == "__main__":
    clean_dataset(SOURCE_DIR, CLEANED_DIR)