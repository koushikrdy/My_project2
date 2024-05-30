from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import pickle

activities_label = {0: "None",1: "Standing still (1 min)",2: "Sitting and relaxing (1 min)",
                    3: "Lying down (1 min)",4: "Walking (1 min)",5: "Climbing stairs (1 min)",
                    6: "Waist bends forward (20x)",7: "Frontal elevation of arms (20x)",
                    8: "Knees bending (crouching) (20x)",9: "Cycling (1 min)",10: "Jogging (1 min)",
                    11: "Running (1 min)",12: "Jump front & back (20x)"
                    }

# Create your views here.
KNN = pickle.load(open("app/model/KNNap.pkl", "rb"))
RF = pickle.load(open("app/model/RFap.pkl", "rb"))

# KNNwithRS = pickle.load(open("app/model/KnnRsAp.pkl", "rb"))
# RFwithRS = pickle.load(open("app/model/RfRsAp.pkl", "rb"))

def index(request):
    return render(request, "index.html")


def login(request):
    return render(request, "login.html")


def abstract(request):
    return render(request, "abstract.html")


def prediction(request):
    if request.method == 'POST':
        file_path = request.FILES['file']
        df = pd.read_csv(file_path)
        data = np.array([df.iloc[0].values])
        print("data:", data)
        # data = RobustScaler().fit(data)
        predicted = KNN.predict(data)[0]
        print("predicted:", predicted)

        labels = activities_label[predicted]
        return render(request, "prediction.html", {'prediction_text': labels})
    return render(request, "prediction.html")


def prediction2(request):
    if request.method == 'POST':
        file_path = request.FILES['file']
        df = pd.read_csv(file_path)
        data = np.array([df.iloc[0].values])
        print(data)
        predicted2 = RF.predict(data)[0]
        print("predicted:", predicted2)

        labels2 = activities_label[predicted2]
        return render(request, "prediction2.html", {'prediction_text2': labels2})
    return render(request, "prediction2.html")


# def prediction3(request):
#     if request.method == 'POST':
#         final_model = request.POST.get("modelname")
#         print("final_model:", final_model)

#         alx = float(request.POST.get("alx"))
#         aly = float(request.POST.get("aly"))
#         alz = float(request.POST.get("alz"))
#         glx = float(request.POST.get("glx"))
#         gly = float(request.POST.get("gly"))
#         glz = float(request.POST.get("glz"))
#         arx = float(request.POST.get("arx"))
#         ary = float(request.POST.get("ary"))
#         arz = float(request.POST.get("arz"))
#         grx = float(request.POST.get("grx"))
#         gry = float(request.POST.get("gry"))
#         grz = float(request.POST.get("grz"))
#         final_features = np.array([[alx, aly, alz, glx, gly, glz, arx, ary, arz, grx, gry, grz]])

#         if final_model == 'RF':
#             final_features_scaled = final_features
#             print("features3:", final_features_scaled)
#             model = RF
#         elif final_model == 'KNN':
#             final_features_scaled = final_features
#             print("features4:", final_features_scaled)
#             model = KNN

#         predicted3 = model.predict(final_features_scaled)
#         print("predicted:", predicted3)

#         labels3 = activities_label[predicted3[0]]
#         return render(request, "prediction3.html", {'prediction_text3': labels3})
#     return render(request, "prediction3.html")

def performance(request):
    return render(request, "performance.html")
