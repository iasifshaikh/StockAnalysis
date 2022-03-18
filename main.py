import pandas as  pd
#import flask_cors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, Response, url_for, redirect
#import Django
#

#from bson import ObjectId

app = Flask(__name__)

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("frontend.html")

@app.route('/predict',methods=['POST','GET'])
def sentiment_analysis():
    if request.method == 'POST':
        inp_1 = request.form['inp_1']
        inp_2 = request.form['inp_2']
        inp_3 = request.form['inp_3']
        inp_4 = request.form['inp_4']
        inp_5 = request.form['inp_5']
        inp_6 = request.form['inp_6']
        inp_7 = request.form['inp_7']
        inp_8 = request.form['inp_8']
        inp_9 = request.form['inp_9']
        inp_10 = request.form['inp_10']
        inp_11 = request.form['inp_11']
        inp_12 = request.form['inp_12']
        inp_13 = request.form['inp_13']
        inp_14 = request.form['inp_14']
        inp_15 = request.form['inp_15']
        inp_16 = request.form['inp_16']


        lst = [inp_1,inp_2,inp_3,inp_4,inp_5,inp_6,inp_7,inp_8,inp_9,inp_9,inp_10,inp_11,inp_12,inp_13,inp_14,inp_15,inp_16]


        df = pd.read_csv('Data.csv', encoding="ISO-8859-1")
        train = df[df['Date'] < '20150101']
        test = df[df['Date'] > '20141231']



        data = train.iloc[:, 2:27]
        data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

        # Renaming column names for ease of access
        list1 = [i for i in range(25)]
        new_Index = [str(i) for i in list1]
        data.columns = new_Index
        data.head(5)

        ## Convertng headlines to lower case
        for index in new_Index:
            data[index] = data[index].str.lower()

        headlines = []
        for row in range(0, len(data.index)):
            headlines.append(' '.join(str(x) for x in data.iloc[row, 0:25]))

        ## implement BAG OF WORDS
        countvector = CountVectorizer(ngram_range=(2, 2))
        traindataset = countvector.fit_transform(headlines)

        # implement RandomForest Classifier
        randomclassifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
        randomclassifier.fit(traindataset, train['Label'])


        data1 = {
            '0': [lst[0]],
            '1': [lst[1]],
            '2': [lst[2]],
            '3': [lst[3]],
            "4": [lst[4]],
            "5": [lst[5]],
            '6': [lst[6]],
            '7': [lst[7]],
            '8': [lst[8]],
            '9': [lst[9]],
            '10': [lst[10]],
            '11': [lst[11]],
            '12': [lst[12]],
            '13': [lst[13]],
            '14': [lst[14]],
            '15':[[lst[15]]]
        }
        dataframe = pd.DataFrame(data1)
        test_transform = []

        for row in range(0, len(dataframe.index)):
            test_transform.append(' '.join(str(x) for x in dataframe.iloc[0, :]))
        test_dataset = countvector.transform(test_transform)
        predictions = randomclassifier.predict(test_dataset)
        output = int(predictions[0])
        op = ''
        if output == 0:
            op = "Stock price will decrease"
        else:
            op = "Stock price will increase"

        return render_template('frontend.html',op_prediction=op)


if __name__ == '__main__':
    app.run(debug=True)