from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from . import forms
from .models import User_SigUp

import os, glob
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevent GUI backend errors
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import joblib

# ========================== AUTH ==========================
def SigUp(request):
    if request.method == 'POST':
        form = forms.User_SigupForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Account Created Successfully')
        else:
            messages.error(request, 'Invalid Credentials')
    return render(request, 'Register.html', {'form': forms.User_SigupForm()})


def UserLogin(request):
    if request.method == 'POST':
        username = request.POST.get('name')
        password = request.POST.get('password')
        try:
            user = User_SigUp.objects.get(Username=username, Password=password)
            if user.Status == 'active':
                return redirect('UserHome')
            else:
                messages.error(request, 'You are not activated yet')
        except:
            messages.error(request, 'Invalid data')
    return render(request, 'UserLogin.html')


def UserHome(request):
    return render(request, 'Users/UserHome.html')


# ========================== DATASET ==========================
class MRI(Dataset):
    def __init__(self):
        tumor, healthy = [], []

        for f in glob.glob(os.path.join(settings.MEDIA_ROOT, 'brain_tumor_dataset/yes/*.jpg')):
            img = cv2.resize(cv2.imread(f), (128, 128))
            img = img.transpose(2,0,1)  # shape: 3x128x128
            tumor.append(img)

        for f in glob.glob(os.path.join(settings.MEDIA_ROOT, 'brain_tumor_dataset/no/*.jpg')):
            img = cv2.resize(cv2.imread(f), (128, 128))
            img = img.transpose(2,0,1)
            healthy.append(img)

        self.images = np.concatenate((np.array(tumor, dtype=np.float32),
                                      np.array(healthy, dtype=np.float32))) / 255.0
        self.labels = np.concatenate((np.ones(len(tumor)), np.zeros(len(healthy))))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            'images': torch.tensor(self.images[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


# ========================== CNN MODEL ==========================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(128*16*16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return torch.sigmoid(x)


# ========================== TRAINING & COMPARISON ==========================
def Traning(request):
    dataset = MRI()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # -------- CNN TRAINING --------
    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    epochs = 30

    model.train()
    for epoch in range(epochs):
        for d in loader:
            optimizer.zero_grad()
            out = model(d['images'])
            loss = loss_fn(out.squeeze(), d['labels'])
            loss.backward()
            optimizer.step()

    # -------- CNN EVALUATION --------
    model.eval()
    outputs, y_true = [], []

    with torch.no_grad():
        for d in loader:
            out = model(d['images']).squeeze().numpy()
            outputs.append(out)
            y_true.append(d['labels'].numpy())

    outputs = np.concatenate(outputs)
    y_true = np.concatenate(y_true)

    cnn_pred = (outputs >= 0.5).astype(int)
    cnn_acc = accuracy_score(y_true, cnn_pred)
    cnn_precision = precision_score(y_true, cnn_pred)
    cnn_recall = recall_score(y_true, cnn_pred)
    cnn_f1 = f1_score(y_true, cnn_pred)

    # -------- FEATURE EXTRACTION FOR ML MODELS --------
    features, labels = [], []
    with torch.no_grad():
        for d in loader:
            x = model.cnn_model(d['images'])
            x = x.view(x.size(0), -1)
            features.append(x.numpy())
            labels.append(d['labels'].numpy())

    X = np.vstack(features)
    y = np.hstack(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------- SVM --------
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_proba = svm.predict_proba(X_test)[:,1]

    # -------- RANDOM FOREST --------
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:,1]

    # -------- LOGISTIC REGRESSION --------
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:,1]

    # -------- ACCURACY METRICS --------
    acc_dict = {
        'CNN': cnn_acc,
        'SVM': accuracy_score(y_test, svm_pred),
        'Random Forest': accuracy_score(y_test, rf_pred),
        'Logistic Regression': accuracy_score(y_test, lr_pred)
    }
    best_model = max(acc_dict, key=acc_dict.get)

    comparison = [
        {'model': 'CNN', 'acc': cnn_acc, 'p': cnn_precision, 'r': cnn_recall, 'f1': cnn_f1},
        {'model': 'SVM', 'acc': accuracy_score(y_test, svm_pred),
         'p': precision_score(y_test, svm_pred),
         'r': recall_score(y_test, svm_pred),
         'f1': f1_score(y_test, svm_pred)},
        {'model': 'Random Forest', 'acc': accuracy_score(y_test, rf_pred),
         'p': precision_score(y_test, rf_pred),
         'r': recall_score(y_test, rf_pred),
         'f1': f1_score(y_test, rf_pred)},
        {'model': 'Logistic Regression', 'acc': accuracy_score(y_test, lr_pred),
         'p': precision_score(y_test, lr_pred),
         'r': recall_score(y_test, lr_pred),
         'f1': f1_score(y_test, lr_pred)}
    ]

    # -------- SAVE MODELS --------
    model_dir = os.path.join(settings.MEDIA_ROOT, 'ml_models')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(svm, os.path.join(model_dir, 'svm.pkl'))
    joblib.dump(rf, os.path.join(model_dir, 'rf.pkl'))
    joblib.dump(lr, os.path.join(model_dir, 'lr.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    torch.save(model.state_dict(), os.path.join(model_dir, 'cnn.pt'))

    # -------- PLOTS: PR & ROC --------
    plots_dir = os.path.join(settings.MEDIA_ROOT, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure()
    precision, recall, _ = precision_recall_curve(y_test, svm_proba)
    plt.plot(recall, precision, label='SVM')
    precision, recall, _ = precision_recall_curve(y_test, rf_proba)
    plt.plot(recall, precision, label='RF')
    precision, recall, _ = precision_recall_curve(y_test, lr_proba)
    plt.plot(recall, precision, label='LR')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    pr_path = os.path.join(plots_dir, 'pr_curve.png')
    plt.savefig(pr_path)
    plt.close()

    plt.figure()
    fpr, tpr, _ = roc_curve(y_test, svm_proba)
    plt.plot(fpr, tpr, label='SVM')
    fpr, tpr, _ = roc_curve(y_test, rf_proba)
    plt.plot(fpr, tpr, label='RF')
    fpr, tpr, _ = roc_curve(y_test, lr_proba)
    plt.plot(fpr, tpr, label='LR')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    roc_path = os.path.join(plots_dir, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()

    # -------- PROGRESS BAR DATA --------
    progress = {
        'cnn': int(cnn_acc*100),
        'svm': int(acc_dict['SVM']*100),
        'rf': int(acc_dict['Random Forest']*100),
        'lr': int(acc_dict['Logistic Regression']*100)
    }

    return render(request, 'Users/UserTraning.html', {
        'comparison': comparison,
        'best_model': best_model,
        'progress': progress,
        'pr_curve': '/media/plots/pr_curve.png',
        'roc_curve': '/media/plots/roc_curve.png'
    })


# ========================== PREDICTION ==========================
# =====================================================
# PREDICTION (CNN / SVM / RF / LR with Confidence)
# =====================================================
# =====================================================
# PREDICTION (CNN / SVM / RF / LR with Confidence)
# ============================================
import os
import torch
import joblib
import numpy as np
from PIL import Image
from django.shortcuts import render
from django.conf import settings
from torchvision import transforms

# --------------------------------------------------
# GLOBAL PATHS
# --------------------------------------------------
CNN_MODEL_PATH = os.path.join(settings.MEDIA_ROOT, "ml_models", "cnn.pt")
SVM_MODEL_PATH = os.path.join(settings.MEDIA_ROOT, "ml_models", "svm.pkl")
RF_MODEL_PATH  = os.path.join(settings.MEDIA_ROOT, "ml_models", "rf.pkl")
LR_MODEL_PATH  = os.path.join(settings.MEDIA_ROOT, "ml_models", "lr.pkl")


CLASS_NAMES = ["No Tumor", "Tumor"]

# --------------------------------------------------
# IMAGE TRANSFORM (MUST MATCH TRAINING)
# --------------------------------------------------
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# --------------------------------------------------
# HELPER: LOAD CNN SAFELY
# --------------------------------------------------
def load_cnn():
    if not os.path.exists(CNN_MODEL_PATH):
        return None
    model = torch.load(CNN_MODEL_PATH, map_location="cpu")
    model.eval()
    return model

# --------------------------------------------------
# HELPER: MEDICAL DECISION LOGIC
# --------------------------------------------------
def medical_decision(prob):
    if prob >= 0.75:
        return "Tumor Detected", "danger"
    elif prob >= 0.55:
        return "Suspicious – Needs Review", "warning"
    else:
        return "No Tumor Detected", "safe"

# --------------------------------------------------
# MAIN PREDICT VIEW
# --------------------------------------------------
# PREDICTION
# =====================================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def load_cnn():
    model = CNN()
    model.load_state_dict(torch.load(os.path.join(settings.MEDIA_ROOT, "ml_models", "cnn.pt"), map_location="cpu"))
    model.eval()
    return model


def predict(request):
    context = {}

    if request.method == "POST" and request.FILES.get("image"):
        img = Image.open(request.FILES["image"]).convert("RGB")
        x = transform(img).unsqueeze(0)

        model_name = request.POST.get("model")
        model = load_cnn()

        with torch.no_grad():
            feat = model.cnn_model(x)
            feat = feat.view(feat.size(0), -1)

        scaler = joblib.load(os.path.join(settings.MEDIA_ROOT, "ml_models", "scaler.pkl"))
        feat = scaler.transform(feat)

        if model_name == "cnn":
            prob = model(x).item()
        elif model_name == "svm":
            prob = joblib.load(os.path.join(settings.MEDIA_ROOT, "ml_models", "svm.pkl")).predict_proba(feat)[0][1]
        elif model_name == "rf":
            prob = joblib.load(os.path.join(settings.MEDIA_ROOT, "ml_models", "rf.pkl")).predict_proba(feat)[0][1]
        else:
            prob = joblib.load(os.path.join(settings.MEDIA_ROOT, "ml_models", "lr.pkl")).predict_proba(feat)[0][1]

        context.update({
            "confidence": round(prob * 100, 2),
            "result": "Tumor Detected" if prob >= 0.5 else "No Tumor Detected",
            "model_used": model_name.upper()
        })

    return render(request, "Users/UserPredict.html", context)
