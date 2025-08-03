## 🧠 Customer Segmentation using K-Means Clustering

This project applies the K-Means clustering algorithm to group customers based on their **Annual Income** and **Spending Score**.  
By identifying clusters like "High Spenders" or "Budget-Conscious Shoppers", businesses can better target their marketing strategies.

---

## 📌 Project Objective

Segment customers into distinct groups based on:
- Annual income (in $1000s)
- Spending behavior

This allows businesses to:
- Understand customer behavior
- Offer personalized promotions
- Enhance marketing efficiency

---

## 🧪 Technologies Used

- **Python**
- NumPy for numerical operations
- Matplotlib for data visualization
- Scikit-learn for applying the KMeans algorithm

---

## 📊 Dataset

The dataset is simulated with 20 customers, each represented by two features:
- **Annual Income**
- **Spending Score**

Example data:
python
[15, 39], [16, 81], [17, 6], [18, 77], [19, 40], ...



## 🧮 K-Means Clustering
We used k=3 clusters to group customers into 3 segments.
The algorithm automatically detects patterns and assigns each customer to one of the clusters.

📈 Visualization
The following chart shows:

[url=https://www.0zz0.com][img]https://www2.0zz0.com/2025/08/03/13/743888741.jpg[/img][/url]

Each customer as a colored point

Cluster centers as red ❌ markers

🔁 How to Run the Code
📥 Clone the repository:

git clone https://github.com/zohoor7/customer-segmentation-kmeans.git

📦 Install the required libraries:
pip install numpy matplotlib scikit-learn

▶️ Run the script:
python kmeans_segmentation.py

🧠 Output Example

Cluster Centers (Annual Income, Spending Score):
Cluster 1: Income = 21.43, Spending Score = 60.43
Cluster 2: Income = 87.50, Spending Score = 75.00
Cluster 3: Income = 83.33, Spending Score = 25.00


📌 Author
Zohoor Awaji
Marketing Specialist & AI Learner
🌐 LinkedIn https://www.linkedin.com/in/zohoor-awaji/

⭐️ If you liked this project, don’t forget to star the repo!
