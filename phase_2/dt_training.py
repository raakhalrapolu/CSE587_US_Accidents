# Decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pylab as plt
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree

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
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=21, stratify=y)

# Instantiate dt_entropy, set 'entropy' as the information criterion
dt_entropy = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)

print(X_train.shape)

# Fit dt_entropy to the training set
dt_entropy.fit(X_train, y_train)

# Use dt_entropy to predict test set labels
y_pred = dt_entropy.predict(X_test)

# Evaluate accuracy_entropy
accuracy_entropy = accuracy_score(y_test, y_pred)

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[1, 2, 3, 4])

cm_display.plot()
plt.show()

# Print accuracy_entropy
print('[Decision Tree -- entropy] accuracy_score: {:.3f}.'.format(accuracy_entropy))

# Instantiate dt_gini, set 'gini' as the information criterion
dt_gini = DecisionTreeClassifier(max_depth=8, criterion='gini', random_state=1)

# Fit dt_entropy to the training set
dt_gini.fit(X_train, y_train)

# Use dt_entropy to predict test set labels
y_pred = dt_gini.predict(X_test)

# Evaluate accuracy_entropy
accuracy_gini = accuracy_score(y_test, y_pred)

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[1, 2, 3, 4])

cm_display.plot()
plt.show()
plt.close()

# Print accuracy_gini
print('[Decision Tree -- gini] accuracy_score: {:.3f}.'.format(accuracy_gini))

# tree visualisation


tree.plot_tree(dt_gini, max_depth=5)
