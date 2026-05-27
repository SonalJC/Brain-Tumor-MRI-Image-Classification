# 🏗️ Infrastructure Damage Detection

A deep learning project that detects structural damage in infrastructure from images using **MobileNetV2** and **Streamlit**.

---

## 📌 About The Project

Identifying infrastructure damage quickly is critical in disaster response and civil engineering. This project uses transfer learning with MobileNetV2 to automatically classify whether a structure is damaged or not — deployed as a real-time web application.

---

## 🎯 Results

| Metric | Score |
|--------|-------|
| 🏋️ Training Accuracy | **94.17%** |
| ✅ Validation Accuracy | **89.17%** |

---

## 🛠️ Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Model:** MobileNetV2 (Transfer Learning + Fine-tuning)
- **Web App:** Streamlit
- **Environment:** Jupyter Notebook
- **Libraries:** NumPy, Matplotlib, OpenCV, Pillow

---

## ✨ Features

- Upload infrastructure images for real-time damage prediction
- Fine-tuned MobileNetV2 with data augmentation for better generalization
- Lightweight model — fast predictions
- Interactive Streamlit web interface

---

## 🚀 How To Run

```bash
# 1. Clone the repository
git clone https://github.com/SonalJC/Infrastructure-damage-detection-streamlit.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

---

## 📂 Project Structure

```
├── app.py                        # Streamlit web application
├── Capstone_project.ipynb        # Jupyter notebook - model training
├── finetuned_damage_model.h5     # Trained model file
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 👩‍💻 Author

**Sonal Chauhan**
- GitHub: [@SonalJC](https://github.com/SonalJC)
- LinkedIn: [sonal-chauhan-457946297](https://www.linkedin.com/in/sonal-chauhan-457946297)
