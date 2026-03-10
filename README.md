# Customer Segmentation using K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Clustering-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![Status](https://img.shields.io/badge/Project-Completed-green)

This project applies **unsupervised machine learning (K-Means Clustering)** to segment customers based on their **annual income and spending behavior**.

Customer segmentation helps businesses understand their customers better and design **targeted marketing strategies** that increase customer engagement and revenue.

The project includes **data analysis, clustering model development, and a Streamlit web application for real-time customer segmentation.**

---

# 🚀 Project Overview

Businesses often have diverse customer groups with different spending habits. Treating all customers the same leads to **inefficient marketing and lower conversions**.

This project builds a **machine learning clustering model** that groups customers into meaningful segments based on their purchasing behavior.

These segments help companies:

- Identify **high-value customers**
- Design **targeted marketing campaigns**
- Improve **customer retention**
- Optimize **marketing budgets**

---

# 📊 Dataset

The dataset contains customer attributes such as:

- Customer ID
- Gender
- Age
- Annual Income
- Spending Score

### Spending Score

The **Spending Score** is a metric assigned by the company based on customer purchasing behavior and spending patterns.

Higher scores indicate **more active and valuable customers**.

---

# 🧠 Machine Learning Workflow

The project follows a structured **Data Science pipeline**.

## 1️⃣ Data Understanding

- Explored dataset structure
- Checked for missing values
- Identified relevant features

---

## 2️⃣ Exploratory Data Analysis (EDA)

Analyzed customer behavior using visualizations:

- Age distribution
- Income distribution
- Spending patterns
- Relationship between income and spending score

EDA helps uncover **hidden patterns in customer purchasing behavior.**

---

## 3️⃣ Feature Selection

Selected the most important features for clustering:

- **Annual Income**
- **Spending Score**

These variables are strong indicators of **customer purchasing behavior**.

---

## 4️⃣ Optimal Number of Clusters

Used the **Elbow Method** to determine the optimal number of clusters.

The elbow point identifies the **best number of customer segments** for meaningful grouping.

---

## 5️⃣ K-Means Clustering

Applied the **K-Means algorithm** to group customers into distinct segments.

Each cluster represents a **unique type of customer behavior**.

---

# 📈 Customer Segments Identified

The clustering algorithm identified several meaningful customer groups.

### High Income – High Spending
Premium customers who generate the **highest revenue**.

### High Income – Low Spending
Customers with high purchasing potential but currently **under-engaged**.

### Low Income – High Spending
Customers who spend actively despite lower income.

### Low Income – Low Spending
Customers with minimal purchasing activity.

---

# 🖥️ Streamlit Web Application

The project includes a **Streamlit application (`app.py`)** that allows users to enter customer details and instantly identify their **customer segment**.

### App Features

✔ Interactive user interface  
✔ Input customer income and spending score  
✔ Real-time customer segmentation  
✔ Instant prediction using trained K-Means model  

---

# 📂 Project Structure


customer-segmentation/
│
├── app.py
│ Streamlit application for customer segmentation
│
├── kmeans_model.pkl
│ Trained K-Means clustering model
│
├── customer-segmentations.ipynb
│ Data analysis and clustering workflow
│
├── requirements.txt
│ Project dependencies
│
└── README.md
│ Project documentation


---

# ⚙️ Installation

Clone the repository:


git clone https://github.com/Vandanasharma-1/Customer-Segmentation

cd customer-segmentation


Install dependencies:


pip install -r requirements.txt


---

# ▶️ Run the Streamlit Application


streamlit run app.py


The application will automatically open in your browser.

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

---

# 📈 Business Impact

Customer segmentation helps businesses:

✔ Identify **high-value customers**  
✔ Design **personalized marketing campaigns**  
✔ Improve **customer engagement**  
✔ Increase **customer lifetime value**  

---

# 👩‍💻 Author

**Vandana Sharma**

Data Scientist | Machine Learning | AI Enthusiast

Passionate about building **data-driven solutions, predictive models, and machine learning systems that solve real-world business problems.**

---

⭐ If you found this project useful, consider giving it a **star**!
