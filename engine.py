from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from datetime import datetime
import os
from pred_model import predict_price
from latlong import get_co
from clf_interior import pred_interior
from clf_exterior import pred_exterior

UPLOAD_FOLDER = 'C:\\Users\\HARSH\\Desktop\\Hackathon\\static\\house_images_user'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:@localhost:3307/bargaining"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class House(db.Model):
    """
    sno, bhk, area, ready, longitude, latitude, description, price, date
    """

    sno = db.Column(db.Integer, primary_key=True)
    bhk = db.Column(db.String(), nullable=False)
    area = db.Column(db.String(), nullable=False)
    ready = db.Column(db.String(), nullable=False)
    longitude = db.Column(db.String(), nullable=False)
    latitude = db.Column(db.String(), nullable=False)
    description = db.Column(db.String(200), nullable=False)
    image_interior = db.Column(db.String(200), nullable=False)
    image_exterior = db.Column(db.String(200), nullable=False)
    price_quoted = db.Column(db.String(), nullable=False)
    price_predicted = db.Column(db.String(), nullable=False)
    date = db.Column(db.String(), nullable=False)


@app.route('/')
def home():
    houses = House.query.filter_by().all()

    return render_template('index.html', houses=houses)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        return render_template('signup.html')


@app.route('/house/<string:sno>', methods=['GET'])
def house_page(sno):
    house_first = House.query.filter_by(sno=sno).first()

    return render_template('post_page.html', house_first=house_first)


@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        bhk = request.form.get('bhk')
        area = request.form.get('area')
        ready_to_move = request.form.get('ready-to-move')
        address = request.form.get('address')
        price = request.form.get('price')
        description = request.form.get('desc')

        # to get lat long from the address
        latitude, longitude = get_co(address)

        # for handling file path
        img_interior = request.files['interior']
        img_exterior = request.files['exterior']

        # saving file in the desired folder

        filename1 = secure_filename(img_interior.filename)
        img_interior.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))

        filename2 = secure_filename(img_exterior.filename)
        img_exterior.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))

        # Now Predicting the price

        d_price = predict_price([[bhk, area, ready_to_move, longitude, latitude]])

        # price of the house after reading the features(data) from the form

        price_pred = int(d_price['price'])

        # Now using image classifiers to classify images(interior)

        if pred_interior(os.path.join(UPLOAD_FOLDER, filename1)) == 1:  # good interior
            flag_price = price_pred * 0.2
            price_pred = price_pred + flag_price

        else:
            flag_price = price_pred * 0.2
            price_pred = price_pred - flag_price

        # Now using image classifiers to classify images(exterior)

        if pred_exterior(os.path.join(UPLOAD_FOLDER, filename2)) == 1:  # good interior
            flag_price = price_pred * 0.2
            price_pred = price_pred + flag_price

        else:
            flag_price = price_pred * 0.2
            price_pred = price_pred - flag_price

        entry = House(bhk=bhk, area=area, ready=ready_to_move, longitude=longitude,
                      latitude=latitude, description=description, image_interior=filename1,
                      image_exterior=filename2, price_quoted=price, price_predicted=int(price_pred), date=datetime.now())

        db.session.add(entry)
        db.session.commit()

        return redirect('/')

    return render_template('form.html')


if __name__ == "__main__":
    app.run(debug=True)
