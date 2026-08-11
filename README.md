🛡️ Anomaly‑Based Intrusion Detection Using Random Forest
This project implements a supervised machine learning Intrusion Detection System (IDS) using the UNSW‑NB15 dataset. The goal is to accurately classify network traffic as normal or malicious using a Random Forest classifier, following a complete pipeline from preprocessing to evaluation and visualization.

📌 Project Overview
The IDS is built using a Random Forest model trained on the UNSW‑NB15 dataset.
The project includes:
• 	Data preprocessing
• 	Feature cleaning and selection
• 	Model training
• 	Evaluation on training and testing datasets
• 	Confusion matrix analysis
• 	Feature importance visualization


📊 Dataset: UNSW‑NB15
• 	Source: Australian Centre for Cyber Security (ACCS)
• 	Files used:
• UNSW_NB15_training-set.csv	
• 	UNSW_NB15_testing-set.csv
• 	Features: 49 attributes + 1 label
• 	Labels:
• 	0 = Normal
• 	1 = Attack
Removed non‑numeric or irrelevant fields:
• 	Id
• 	attack_cat
• 	proto
• 	service
• 	state


🧠 Machine Learning Model
Algorithm: Random Forest Classifier
Library: scikit‑learn
Parameters:
• 	n_estimators = 100
• 	random_state = 42


🔧 Preprocessing Steps
• 	Loaded training and testing datasets using pandas
• 	Dropped categorical and unused columns
• 	Removed missing/null values
• 	Split data into features (X) and labels (y)


📈 Model Performance
Training Results
• 	~98% accuracy
• 	Balanced precision and recall
• 	Minimal overfitting

Testing Results
• 	~90% accuracy
• 	High recall for attack detection
• 	Slightly lower precision for normal traffic


📊 Feature Importance
Top contributing features:
• 	ct_dst_src_ltm
• 	std
• 	ct_dst_sport_ltm
• 	sbytes
• 	ct_state_ttl

These features strongly influence anomaly detection and help distinguish malicious behavior.

🖥️ Technologies Used
• 	Python 3
• 	Pandas
• 	Scikit‑learn
• 	Seaborn
• 	Matplotlib
• 	Linux
• 	Vim

🚀 Project Highlights
• 	Built a complete supervised IDS pipeline
• 	Achieved strong generalization on real testing data
• 	Identified critical features for anomaly detection
• 	Gained hands‑on experience in cybersecurity analytics


🏁 Conclusion
This project strengthened my skills in machine learning, data preprocessing, and cybersecurity analytics.
The Random Forest‑based IDS shows strong performance and serves as a solid foundation for future enhancements such as:
• 	Hyperparameter tuning
• 	Real‑time detection
• 	Deployment as a security tool
• 	Integration with SIEM or SOC workflows
