import os
import requests
from flask import Flask, escape, request, render_template, session
from src.add_noise import add_noise_one
from src.prediction_denoise import predictOne
import random
import sqlite3

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("home.html")

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup.html")


@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template("index.html")
    
@app.route('/predict', methods=['GET', 'POST'])
def predict():   
    if 'filename' in session:
        #os.remove('static/uploaded/' + session['filename'])
        session.pop('filename')
    if 'noise_filename' in session:
        #os.remove('static/uploaded/' + session['noise_filename'])
        session.pop('noise_filename')
    if 'denoise_filename' in session:
        #os.remove('static/uploaded/' + session['denoise_filename'])
        session.pop('denoise_filename')
    return render_template('index.html')

@app.route('/upload', methods=['POST','GET'])
def uploadFile():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        print(filename)
        file.save('static/uploaded/'+filename)
        session['filename'] = filename
        if 'noise_filename' in session:
            #os.remove('static/uploaded/' + session['noise_filename'])
            session.pop('noise_filename')
        return render_template('index.html', filename=filename)

@app.route('/addnoise<string:id>')
def addNoise(id):
    if 'filename' not in session:
        return index()
    if os.path.exists('static/uploaded/' + 'denoise_' + session['filename']):
        os.remove('static/uploaded/' + 'denoise_' + session['filename'])
    if 'noise_filename' in session:
        os.remove('static/uploaded/' + session['noise_filename'])
        session.pop('noise_filename')
    #if 'noise_filename' not in session:
    noise_file = random.choice(os.listdir('data/noise/' + id))
    print('noise_file:',id,noise_file)
    add_noise_one('static/uploaded/' + session['filename'], 'data/noise/{}/{}'.format(id,noise_file),'static/uploaded/noise_' + id + session['filename'])
    session['noise_filename'] = 'noise_' + id + session['filename']
    return render_template('index.html', filename=session['filename'], noise_filename=session['noise_filename'])


@app.route('/denoise')
def removeNoise():
    if 'filename' not in session:
        return index()
    if 'noise_filename' not in session:
        return index()
    predictOne('static/uploaded/' + session['noise_filename'], 'static/uploaded/denoise_' + session['filename'])
    session['denoise_filename'] = 'denoise_' + session['filename']
    #session.pop('noise_filename')
    return render_template('index.html', filename=session['filename'], noise_filename=session['noise_filename'], denoise_filename=session['denoise_filename'])

if __name__ == "__main__":
    app.debug = True
    app.secret_key = 'dangvansam'
    app.run()