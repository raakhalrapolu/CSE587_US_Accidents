# Random Forest algorithm
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from Phase_1.preprocess import accidents_data
from lr_training import feature_list
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

accidents = accidents_data
df_sel = accidents[feature_list].copy()
df_sel.info()

df2 = df_sel.dropna()

df_sel.isnull().mean()

Years_list = [2016., 2017., 2021., 2020., 2019., 2018.]

df_y1 = df2[df2['Year'] == 2020.0]
df_y1.drop('Year', axis=1, inplace=True)
df_y1.info()

# Set the target for the prediction
target = 'Severity'

# Create arrays for the features and the response variable

# set X and y
y = df_y1[target]
X = df_y1.drop(target, axis=1)

# Split the data set into training and testing data sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

scaler = StandardScaler()
X_res = scaler.fit_transform(X_res)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=21, stratify=y)

# Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)

# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Get the accuracy score
acc = accuracy_score(y_test, y_pred)

# Append to the accuracy list
# accuracy_lst.append(acc)


# Model Accuracy, how often is the classifier correct?
print("[Randon forest algorithm] accuracy_score: {:.3f}.".format(acc))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[1, 2, 3, 4])

cm_display.plot()
plt.show()
plt.close()

feature_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Creating a bar plot, displaying only the top k features
k = 10
sns.barplot(x=feature_imp[:10], y=feature_imp.index[:k])
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
