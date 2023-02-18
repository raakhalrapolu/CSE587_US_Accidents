import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from Phase_1.preprocess import accidents_data
from lr_training import feature_list
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint

filepath = 'my_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

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

y_test = y_test - 1
y_train = y_train - 1

inputs = tf.keras.Input(shape=(X.shape[1],))
x = tf.keras.layers.Dense(256, activation='relu')(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(4, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

batch_size = 20
epochs = 100

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[checkpoint,
               tf.keras.callbacks.ReduceLROnPlateau(),
               tf.keras.callbacks.EarlyStopping(
                   monitor='val_loss',
                   patience=3,
                   restore_best_weights=True
               )
               ]
)

y_test = y_test - 1

print("Test Accuracy:", model.evaluate(X_test, y_test, verbose=0)[1])


