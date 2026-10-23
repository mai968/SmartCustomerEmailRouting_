
# from pathlib import Path

# import joblib
# import streamlit as st


# # Đường dẫn đến mô hình đã lưu
# BASE_DIR = Path(__file__).resolve().parent.parent

# MODEL_PATH = BASE_DIR / "models" / "linear_svm.joblib"
# VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.joblib"


# # Cấu hình giao diện
# st.set_page_config(
#     page_title="Email Routing Demo",
#     page_icon="📩",
#     layout="centered",
# )

# st.title("📩 Smart Customer Email Routing")
# st.write(
#     "Nhập nội dung email để mô hình Linear SVM "
#     "dự đoán nhóm hỗ trợ phù hợp."
# )


# # Tải mô hình và vectorizer
# @st.cache_resource
# def load_artifacts():
#     model = joblib.load(MODEL_PATH)
#     vectorizer = joblib.load(VECTORIZER_PATH)
#     return model, vectorizer


# model, vectorizer = load_artifacts()


# # Form nhập email
# with st.form("email_form"):
#     subject = st.text_input(
#         "Subject (Tiêu đề email)",
#         placeholder="Ví dụ: Unable to access my account",
#     )

#     body = st.text_area(
#         "Body (Nội dung email)",
#         placeholder="Nhập nội dung email tại đây...",
#         height=200,
#     )

#     submitted = st.form_submit_button("Dự đoán queue")


# # Dự đoán khi người dùng gửi form
# if submitted:
#     if not subject.strip() and not body.strip():
#         st.warning("Vui lòng nhập tiêu đề hoặc nội dung email.")
#     else:
#         email_text = f"{subject.strip()} {body.strip()}"

#         email_tfidf = vectorizer.transform([email_text])
#         predicted_queue = model.predict(email_tfidf)[0]

#         st.subheader("Kết quả dự đoán")
#         st.success(f"Queue được dự đoán: **{predicted_queue}**")

#         st.caption(
#             "Đây là dự đoán của mô hình, "
#             "không phải xác nhận chắc chắn về nhóm xử lý."
#         )


from pathlib import Path

import joblib
import numpy as np
import streamlit as st


# Đường dẫn đến mô hình đã lưu
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "linear_svm.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.joblib"

TOP_K = 3  # số queue gợi ý hiển thị


# Cấu hình giao diện
st.set_page_config(
    page_title="Email Routing Chat",
    page_icon="📩",
    layout="centered",
)

st.title("📩 Smart Customer Email Routing")
st.caption(
    "Dán nội dung email vào ô chat bên dưới, "
    "mô hình Linear SVM sẽ dự đoán nhóm hỗ trợ phù hợp."
)


# Tải mô hình và vectorizer
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


model, vectorizer = load_artifacts()


def predict_queue(email_text: str):
    """Trả về (queue dự đoán, danh sách top-k [(queue, điểm)])."""
    email_tfidf = vectorizer.transform([email_text])
    predicted = model.predict(email_tfidf)[0]

    scores = model.decision_function(email_tfidf)
    classes = model.classes_

    # Bài toán nhị phân: decision_function trả về mảng 1 chiều
    if scores.ndim == 1:
        scores = np.array([-scores[0], scores[0]])
    else:
        scores = scores[0]

    top_idx = np.argsort(scores)[::-1][:TOP_K]
    top = [(classes[i], float(scores[i])) for i in top_idx]
    return predicted, top


def build_reply(predicted, top) -> str:
    lines = [f"Queue được dự đoán: **{predicted}**", ""]

    if len(top) > 1:
        lines.append(f"Top {len(top)} nhóm có điểm cao nhất (decision score):")
        for queue, score in top:
            lines.append(f"- {queue}: `{score:.3f}`")
        lines.append("")

    lines.append(
        "_Đây là dự đoán của mô hình, "
        "không phải xác nhận chắc chắn về nhóm xử lý._"
    )
    return "\n".join(lines)


# Khởi tạo lịch sử chat
WELCOME = (
    "Xin chào! Hãy dán nội dung email của khách hàng "
    "(có thể gồm cả tiêu đề và nội dung), mình sẽ gợi ý queue phù hợp."
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME}]


# Nút xóa lịch sử ở sidebar
with st.sidebar:
    if st.button("🗑️ Xóa cuộc trò chuyện"):
        st.session_state.messages = [{"role": "assistant", "content": WELCOME}]
        st.rerun()


# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Nhận tin nhắn mới
user_input = st.chat_input("Nhập hoặc dán nội dung email...")

if user_input:
    # Hiển thị và lưu tin nhắn của người dùng
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Dự đoán và trả lời
    with st.chat_message("assistant"):
        if not user_input.strip():
            reply = "Vui lòng nhập nội dung email."
        else:
            with st.spinner("Đang phân tích email..."):
                predicted, top = predict_queue(user_input.strip())
            reply = build_reply(predicted, top)
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})