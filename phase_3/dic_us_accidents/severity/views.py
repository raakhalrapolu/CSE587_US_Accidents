from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
from common import file_storage_utility
from severity import admin
from severity.services import pred_services
import pickle
import traceback
import tensorflow as tf
from dic_us_accidents.settings import BASE_DIR

scaler_path = 'resources/scaler.pkl'
scaler = pickle.load(open(scaler_path, 'rb'))
std_path = 'resources/std_scaler.pkl'
scaler_std = pickle.load(open(std_path, 'rb'))
model_path = 'resources/my_best_model.epoch08-loss0.381.hdf5'
saved_model = tf.keras.models.load_model(
    model_path, custom_objects=None, compile=True, options=None)


# Create your views here.

class PredictionDetails(APIView):
    @staticmethod
    def post(request):
        try:
            wind_speed = request.data['wind_speed']
            pressure = request.data['pressure']
            humidity = request.data['humidity']
            visibility = request.data['visibility']
            temperature = request.data['temperature']
            traffic_signal = request.data['traffic_signal']
            crossing = request.data['crossing']
            junction = request.data['junction']

            if isinstance(temperature, float):
                temperature = temperature
            if isinstance(humidity, float):
                humidity = humidity
            if isinstance(pressure, float):
                pressure = pressure
            if isinstance(visibility, float):
                visibility = visibility
            if isinstance(wind_speed, float):
                wind_speed = wind_speed
            if isinstance(traffic_signal, int):
                traffic_signal = traffic_signal
            if isinstance(crossing, int):
                crossing = crossing
            if isinstance(junction, int):
                junction = junction

            input_array = [float(wind_speed), float(pressure), float(humidity), float(visibility), float(temperature),
                           int(traffic_signal), int(crossing), int(junction)]
            output_pred = pred_services.severity_pred(input_array, scaler, scaler_std, saved_model)
            return Response({'result': output_pred}, status=200)
        except Exception as e:
            traceback.format_exc()
            return Response({"message": "Exception Error " + str(e)})
