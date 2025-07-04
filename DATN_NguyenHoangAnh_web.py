import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Load mô hình và dữ liệu mẫu
# Tải mô hình
with open('RandomForest_Model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
sample_data = pd.read_csv('Thyroid_Diff.csv')

# Các cột phân loại
categorical_cols = ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy',
                    'Thyroid Function', 'Physical Examination', 'Adenopathy', 'Pathology',
                    'Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response']

# Tạo LabelEncoder cho các cột phân loại
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(sample_data[col].astype(str))
    label_encoders[col] = le

# ===== Mapping tiếng Việt =====

gender_mapping = {"F": "Nữ", "M": "Nam"}
smoking_mapping = {"No": "Không", "Yes": "Có"}
hx_smoking_mapping = {"No": "Không", "Yes": "Có"}
hx_radiotherapy_mapping = {"No": "Không", "Yes": "Có"}

thyroid_function_mapping = {
    "Euthyroid": "Bình giáp",
    "Clinical Hyperthyroidism": "Bệnh cường giáp lâm sàng",
    "Clinical Hypothyroidism": "Suy giáp lâm sàng",
    "Subclinical Hyperthyroidism": "Cường giáp dưới lâm sàng",
    "Subclinical Hypothyroidism": "Suy giáp dưới lâm sàng"
}

physical_exam_mapping = {
    "Single nodular goiter-left": "Bướu cổ đơn nhân bên trái",
    "Multinodular goiter": "Bướu cổ đa nhân",
    "Single nodular goiter-right": "Bướu cổ đơn nhân bên phải",
    "Normal": "Bình thường",
    "Diffuse goiter": "Bướu cổ lan tỏa"
}

adenopathy_mapping = {
    "No": "Không",
    "Right": "Phải",
    "Extensive": "Lan rộng",
    "Left": "Trái",
    "Bilateral": "Hai bên",
    "Posterior": "Sau"
}

pathology_mapping = {
    "Micropapillary": "Vi nhú",
    "Papillary": "Nhú",
    "Follicular": "Nang trứng",
    "Hurthel cell": "Tế bào Hurthle"
}

focality_mapping = {
    "Uni-Focal": "1 điểm",
    "Multi-Focal": "Đa điểm"
}

risk_mapping = {
    "Low": "Rủi ro thấp",
    "Intermediate": "Rủi ro mức trung",
    "High": "Rủi ro cao"
}

N_mapping = {
    "N0": "Không có hạch bạch huyết khu vực nào bị ảnh hưởng",
    "N1a": "Ảnh hưởng tối thiểu của hạch bạch huyết khu vực",
    "N1b": "Ảnh hưởng rộng rãi của hạch bạch huyết khu vực"
}

M_mapping = {
    "M0": "Không có di căn xa",
    "M1": "Có di căn xa"
}

response_mapping = {
    "Indeterminate": "Không xác định",
    "Excellent": "Xuất sắc",
    "Structural Incomplete": "Cấu trúc chưa hoàn thiện",
    "Biochemical Incomplete": "Sinh hóa chưa hoàn thiện"
}

# Hàm ánh xạ ngược (việt → anh)
def reverse_lookup(mapping_dict, selected_value):
    return next(k for k, v in mapping_dict.items() if v == selected_value)

# ================= Giao diện Streamlit =================

st.title("Dự đoán tái phát ung thư tuyến giáp")
st.markdown("Vui lòng nhập thông tin bệnh nhân:")

age = st.number_input("Độ tuổi", min_value=15, max_value=82, value=30)

gender_vi = st.selectbox("Giới tính", list(gender_mapping.values()))
gender = reverse_lookup(gender_mapping, gender_vi)

smoking_vi = st.selectbox("Hút thuốc", list(smoking_mapping.values()))
smoking = reverse_lookup(smoking_mapping, smoking_vi)

hx_smoking_vi = st.selectbox("Tiền sử hút thuốc", list(hx_smoking_mapping.values()))
hx_smoking = reverse_lookup(hx_smoking_mapping, hx_smoking_vi)

hx_radiotherapy_vi = st.selectbox("Tiền sử xạ trị", list(hx_radiotherapy_mapping.values()))
hx_radiotherapy = reverse_lookup(hx_radiotherapy_mapping, hx_radiotherapy_vi)

thyroid_vi = st.selectbox("Chức năng tuyến giáp", list(thyroid_function_mapping.values()))
thyroid_func = reverse_lookup(thyroid_function_mapping, thyroid_vi)

physical_vi = st.selectbox("Khám lâm sàng", list(physical_exam_mapping.values()))
physical_exam = reverse_lookup(physical_exam_mapping, physical_vi)

adenopathy_vi = st.selectbox("Hạch bạch huyết cổ", list(adenopathy_mapping.values()))
adenopathy = reverse_lookup(adenopathy_mapping, adenopathy_vi)

pathology_vi = st.selectbox("Mô bệnh học", list(pathology_mapping.values()))
pathology = reverse_lookup(pathology_mapping, pathology_vi)

focality_vi = st.selectbox("Đa ổ / Đơn ổ", list(focality_mapping.values()))
focality = reverse_lookup(focality_mapping, focality_vi)

risk_vi = st.selectbox("Mức độ nguy cơ", list(risk_mapping.values()))
risk = reverse_lookup(risk_mapping, risk_vi)

# Lọc bỏ NaN trong T
T_options = sample_data["T"].dropna().unique()
T = st.selectbox("Phân loại khối u (T)", T_options)

st.markdown("""
**📘 Giải thích phân loại T:**
- **T1a**: Khối u ≤ 1 cm, nằm hoàn toàn trong tuyến giáp  
- **T1b**: Khối u > 1 cm nhưng ≤ 2 cm  
- **T2**: Khối u > 2 cm nhưng ≤ 4 cm  
- **T3a**: Khối u > 4 cm nhưng chưa xâm lấn  
- **T3b**: Khối u phát triển ra ngoài tuyến giáp  
- **T4a**: Xâm lấn các cấu trúc gần như khí quản, thực quản  
- **T4b**: Xâm lấn sâu vào cơ quan quan trọng, không thể phẫu thuật  
""")

N_vi = st.selectbox("Phân loại hạch (N)", list(N_mapping.values()))
N = reverse_lookup(N_mapping, N_vi)

M_vi = st.selectbox("Phân loại di căn xa (M)", list(M_mapping.values()))
M = reverse_lookup(M_mapping, M_vi)

stage = st.selectbox("Giai đoạn bệnh", sample_data["Stage"].dropna().unique())

response_vi = st.selectbox("Phản ứng với điều trị", list(response_mapping.values()))
response = reverse_lookup(response_mapping, response_vi)

# ================== DỰ ĐOÁN ==================

if st.button("🔍 Dự đoán"):
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Smoking': smoking,
        'Hx Smoking': hx_smoking,
        'Hx Radiothreapy': hx_radiotherapy,
        'Thyroid Function': thyroid_func,
        'Physical Examination': physical_exam,
        'Adenopathy': adenopathy,
        'Pathology': pathology,
        'Focality': focality,
        'Risk': risk,
        'T': T,
        'N': N,
        'M': M,
        'Stage': stage,
        'Response': response
    }])

    # Mã hóa dữ liệu
    for col in categorical_cols:
        input_data[col] = label_encoders[col].transform(input_data[col].astype(str))

    # Dự đoán
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    # Hiển thị kết quả
    if prediction == 1:
        st.error("⚠️ Kết quả: **CÓ TÁI PHÁT**\n\nNguy cơ tái phát cao, cần theo dõi sát.")
    else:
        st.success("✅ Kết quả: **KHÔNG TÁI PHÁT**\n\nNguy cơ tái phát thấp, tiếp tục theo dõi định kỳ.")
