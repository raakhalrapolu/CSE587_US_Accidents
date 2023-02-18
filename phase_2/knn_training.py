import matplotlib.pylab as plt
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from Phase_1.preprocess import accidents_data
from lr_training import feature_list

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


# KNN training

def knn_training(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)

    knn = KNeighborsClassifier(n_neighbors=6)

    # Fit the classifier to the data
    knn.fit(X_train, y_train)

    # Predict the labels for the training data X
    y_pred = knn.predict(X_test)

    # Get the accuracy score
    acc = accuracy_score(y_test, y_pred)

    print('[K-Nearest Neighbors (KNN)] knn.score: {:.3f}.'.format(knn.score(X_test, y_test)))
    print('[K-Nearest Neighbors (KNN)] accuracy_score: {:.3f}.'.format(acc))
    return acc, y_pred, y_test


# training on upsampled data sample year 2020

acc, y_pred, y_test = knn_training(X_res, y_res)

conf = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=[1, 2, 3, 4])

cm_display.plot()
plt.show()
plt.close()