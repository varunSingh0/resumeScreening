import pandas as pd
from tkinter import *
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import re
te = StringVar


def visualize():
    data = pd.read_csv("UpdatedResumeDataset.csv")
    plt.figure(figsize=(20, 15))
    sns.countplot(y="Category", data=data)
    count = data['Category'].value_counts()
    label = data['Category'].value_counts().keys()

    plt.figure(figsize=(200, 200))

    pie = plt.pie(count, labels=label, autopct="%1.2f%%")
    plt.show()


def read_dataset():
    temp1 = Tk()
    temp1.geometry("1920x1080")
    te = pd.read_csv('UpdatedResumeDataset.csv')
    sb = Scrollbar(temp1, orient=VERTICAL)
    sb.pack(side=RIGHT, fill="y")
    sb2 = Scrollbar(temp1, orient=HORIZONTAL)
    sb2.pack(side=BOTTOM, fill="x")
    l = Listbox(temp1, width=350, height=300,
                yscrollcommand=sb.set, xscrollcommand=sb2.set)
    l.config(yscrollcommand=sb.set)
    sb.config(command=l.yview)
    l.config(xscrollcommand=sb2.set)
    sb2.config(command=l.xview)
    l.pack(pady=15)
    l.insert(1, "Category                              Resume")
    for i in range(2, 964):
        s = "%s                          %s" % (
            te["Category"][i-2], te["Resume"][i-2])
        l.insert(i, s)
    temp1.mainloop()


def clean_text(text):
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', '', text)
    text = re.sub('[%s]' % re.escape(
        """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    return text


def clean():
    data = pd.read_csv("UpdatedResumeDataset.csv")
    data['cleaned_resume'] = data.Resume.apply(lambda x: clean_text(x))
    temp2 = Tk()
    temp2.geometry("1000x1300")
    sb = Scrollbar(temp2, orient=VERTICAL)
    sb.pack(side=RIGHT, fill="y")
    sb2 = Scrollbar(temp2, orient=HORIZONTAL)
    sb2.pack(side=BOTTOM, fill="x")
    l = Listbox(temp2, width=350, height=300,
                yscrollcommand=sb.set, xscrollcommand=sb2.set)
    l.config(yscrollcommand=sb.set)
    sb.config(command=l.yview)
    l.config(xscrollcommand=sb2.set)
    sb2.config(command=l.xview)
    l.pack(pady=15)
    l.insert(1,
             "Category                                                       Resume                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          Cleaned_resume")
    for i in range(2, 964):
        s = "%s%s%s" % (data["Category"][i - 2], data["cleaned_resume"][i - 2], data['cleaned_resume'][i - 2])
        l.insert(i, s)
    temp2.mainloop()


def final_processing():
    data = pd.read_csv("UpdatedResumeDataset.csv")
    data['cleaned_resume'] = data.Resume.apply(lambda x: clean_text(x))

    var_mod = ['Category']
    le = LabelEncoder()
    for i in var_mod:
        data[i] = le.fit_transform(data[i])
    requiredText = data['cleaned_resume'].values
    requiredTarget = data['Category'].values
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words='english',
        max_features=1500)
    word_vectorizer.fit(requiredText)
    WordFeatures = word_vectorizer.transform(requiredText)
    # Model(ML)
    X_train, X_test, y_train, y_test = train_test_split(
        WordFeatures, requiredTarget, random_state=0, test_size=0.2)
    print(X_train.shape)
    print(X_test.shape)
    clf = OneVsRestClassifier(KNeighborsClassifier())
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    # Results
    print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(
        clf.score(X_train, y_train)))
    print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(
        clf.score(X_test, y_test)))
    print("n Classification report for classifier %s:n%sn" %
          (clf, metrics.classification_report(y_test, prediction)))


sakshaat = Tk()
sakshaat.geometry("1920x1080")
load = Image.open('immmage.jpg')
render = ImageTk.PhotoImage(load)
image = Label(sakshaat, image=render)
image.place(x=0, y=0)
resume_scr = Label(sakshaat, text="Resume Screening", height=5,
                   width=25, bg='dark salmon', font=('Arial', 20)).place(x=710, y=0)

b = Button(sakshaat, bg='dark salmon', fg='black', font=('Arial', 17),
           height=6, width=25, text="Show Dataset", command=read_dataset)
b.place(x=10, y=0)

b2 = Button(sakshaat, bg='dark salmon', fg='black', font=(
    'Arial', 17), height=6, width=25, text="Clean Dataaset", command=clean)
b2.place(x=10, y=650)

b3 = Button(sakshaat, bg='dark salmon', fg='black', font=('Arial', 17),
            height=6, width=25, text="Graphical Representation", command=visualize)
b3.place(x=1500, y=0)

b4 = Button(sakshaat, bg='dark salmon', fg='black', font=('Arial', 17),
            height=6, width=25, text='Final Output', command=final_processing)
b4.place(x=1500, y=650)

sakshaat.mainloop()
