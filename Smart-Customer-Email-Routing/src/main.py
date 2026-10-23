
from pathlib import Path
import pandas as pd

# Đường dẫn đến dataset
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
)

def main():
    print("Duong dan dang tim:", DATA_PATH)
    print("File ton tai:", DATA_PATH.exists())
    
    # 1. Kiểm tra file có tồn tại không
    if not DATA_PATH.exists():
        print(f"KHONG TIM THAY FILE: {DATA_PATH}")
        print("Hay kiem tra ten file va vi tri trong thu muc data/.")
        return

    print("Dang doc dataset...")

    # 2. Đọc CSV
    # sep=None và engine="python" giúp tự nhận diện dấu phân cách phổ biến.
    # Nếu dataset sử dụng dấu phẩy chuẩn, pandas vẫn đọc được bình thường.
    df = pd.read_csv(
        DATA_PATH,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
        on_bad_lines="error",
    )

    # 3. Hiển thị thông tin cơ bản
    print("\n===== KICH THUOC DATASET =====")
    print(f"So dong: {df.shape[0]:,}")
    print(f"So cot: {df.shape[1]}")

    print("\n===== TEN CAC COT =====")
    print(df.columns.tolist())

    print("\n===== 5 DONG DAU TIEN =====")
    print(df.head().to_string())

    print("\n===== KIEU DU LIEU =====")
    print(df.dtypes.to_string())

    print("\n===== SO LUONG GIA TRI THIEU =====")
    print(df.isna().sum().to_string())

    print("\n===== TY LE TRUNG LAP =====")
    print(f"So dong trung lap: {df.duplicated().sum():,}")

    print("\nDa doc dataset thanh cong.")


if __name__ == "__main__":
    main()

# .\.venv\Scripts\Activate.ps1   
# lệnh vào mt ảo