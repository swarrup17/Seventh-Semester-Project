import uuid
from flask import Flask, request, render_template, session, url_for, redirect
import random
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
import pandas as pd
import math
import hmac
import hashlib
import base64

# object of flask
# built in syntax cannot change
app = Flask(__name__)

# for database mysqlclient and flasksqlalchemy is installed
# database configuration
app.secret_key = "secretkeycanbeanythingonlyforsecuritypurpose"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:@localhost:3306/ecom?ssl_disabled=true"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# ----------------------
# Models 
# ----------------------
class DisplayProduct(db.Model):
    __tablename__ = 'displayproduct'

    pid = db.Column(db.Integer, primary_key=True, autoincrement=True)
    pname = db.Column(db.String(255), nullable=False)
    reviewcount = db.Column(db.Float, nullable=False)
    brand = db.Column(db.String(255), nullable=False)
    imageurl = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Float, nullable=False)
    description = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(255), nullable=False)
    price = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f'<DisplayProduct {self.pname}>'


class Products(db.Model):
    __tablename__ = 'products'

    ID = db.Column(db.Integer, primary_key=True, autoincrement=True)
    productId = db.Column(db.String(255), nullable=False)
    productname = db.Column(db.String(255), nullable=False)
    reviewcount = db.Column(db.Float, nullable=False)
    productbrand = db.Column(db.String(255), nullable=False)
    imageurl = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Float, nullable=False)
    description = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(255), nullable=False)
    price = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f'<Products {self.productname}>'


class Signup(db.Model):
    __tablename__ = 'signup'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    repassword = db.Column(db.String(255), nullable=False)
    status = db.Column(db.Integer, nullable=False)
    role = db.Column(db.String(255), nullable=False)

    carts = db.relationship('Cart', back_populates='user', cascade="all, delete-orphan")
    purchases = db.relationship('Purchase', back_populates='user', cascade="all, delete-orphan")


class Signin(db.Model):
    __tablename__ = 'signin'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    status = db.Column(db.Integer, nullable=False)


class Admin(db.Model):
    __tablename__ = 'admin'
    id = db.Column(db.Integer, primary_key=True)
    adminName = db.Column(db.String(255), nullable=False)
    adminPassword = db.Column(db.String(255), nullable=False)
    adminRepassword = db.Column(db.String(255), nullable=True)
    role = db.Column(db.String(255), nullable=False)


class Cart(db.Model):
    __tablename__ = 'carts'
    cartid = db.Column(db.Integer, primary_key=True, autoincrement=True)
    userid = db.Column(db.Integer, db.ForeignKey('signup.id'), nullable=False)
    productid = db.Column(db.String(255), nullable=False)
    productname = db.Column(db.String(255), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    image = db.Column(db.String(255), nullable=False)
    price = db.Column(db.Float, nullable=False)

    user = db.relationship('Signup', back_populates='carts')


class Purchase(db.Model):
    __tablename__ = 'purchase'
    purchaseid = db.Column(db.Integer, primary_key=True, autoincrement=True)
    productid = db.Column(db.String(255), nullable=False)
    productname = db.Column(db.String(255), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    productprice = db.Column(db.Float, nullable=False)
    purchaseTime = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())
    userid = db.Column(db.Integer, db.ForeignKey('signup.id'), nullable=False)

    user = db.relationship('Signup', back_populates='purchases')


class Category(db.Model):
    __tablename__ = 'category'
    id = db.Column(db.Integer, primary_key=True)
    Categories = db.Column(db.String(255), nullable=False)


# ----------------------
# Load data for recommendations
# ----------------------
# Ensure these CSV files exist in your project
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_dataset.csv")

import pandas as pd

csv_file = "models/clean_dataset.csv"

try:
    # Use engine='python' to handle irregular CSVs
    # on_bad_lines='skip' will skip problematic rows
    train_data = pd.read_csv(csv_file, engine='python', on_bad_lines='skip', quotechar='"')

    # Optional: reset the index after skipping rows
    train_data.reset_index(drop=True, inplace=True)

    print(f"CSV loaded successfully. Number of rows: {len(train_data)}")

except FileNotFoundError:
    print(f"Error: File {csv_file} not found.")
    train_data = pd.DataFrame()  # Empty DataFrame as fallback

except Exception as e:
    print(f"Error loading CSV: {e}")
    train_data = pd.DataFrame()  # Empty DataFrame as fallback

# ----------------------
# Helper / util functions
# ----------------------
def gen_sha256(key, message):
    key = key.encode('utf-8')
    message = message.encode('utf-8')
    hmac_sha256 = hmac.new(key, message, hashlib.sha256)
    signature = base64.b64encode(hmac_sha256.digest()).decode('utf-8')
    return signature


def truncate(text, length):
    if text is None:
        return ""
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text


def compute_tf_idf(train_data):
    term_freqs = []
    doc_freqs = {}
    num_documents = len(train_data)

    for tags in train_data['Tags'].fillna(''):
        terms = tags.lower().split()
        term_count = {}
        for term in terms:
            term_count[term] = term_count.get(term, 0) + 1
        term_freqs.append(term_count)
        for term in term_count.keys():
            doc_freqs[term] = doc_freqs.get(term, 0) + 1

    tf_idf_matrix = []
    for term_count in term_freqs:
        tf_idf_vector = {}
        for term, count in term_count.items():
            tf = count / len(term_count) if len(term_count) > 0 else 0
            idf = math.log(num_documents / (1 + doc_freqs.get(term, 0))) if num_documents > 0 else 0
            tf_idf_vector[term] = tf * idf
        tf_idf_matrix.append(tf_idf_vector)
    return tf_idf_matrix


def cosine_similarity(vector1, vector2):
    dot_product = sum(vector1.get(term, 0) * vector2.get(term, 0) for term in vector1)
    norm1 = math.sqrt(sum(v ** 2 for v in vector1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vector2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def content_based_recommendations(train_data, item_name, top_n=10):
    if item_name not in train_data['Name'].values:
        return pd.DataFrame()

    tf_idf_matrix = compute_tf_idf(train_data)
    item_index = train_data[train_data['Name'] == item_name].index[0]
    similarities = []
    for i, tf_idf_vector in enumerate(tf_idf_matrix):
        similarity = cosine_similarity(tf_idf_matrix[item_index], tf_idf_vector)
        similarities.append((i, similarity))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    top_similar_indices = [idx for idx, _ in similarities[1:top_n + 1]]
    recommended_items_details = train_data.iloc[top_similar_indices][
        ['ID', 'Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating', 'Price', 'Description']]
    return recommended_items_details


# ----------------------
# Routes
# ----------------------
@app.route("/")
def index():
    products = DisplayProduct.query.all()
    return render_template('index.html', data=products)


@app.route("/main")
def main():
    return render_template('main.html', content_based_rec=None, truncate=truncate)


@app.route("/index")
def indexredirect():
    products = DisplayProduct.query.all()
    return render_template('index.html', data=products)


@app.route("/fetch")
def fetch():
    products = DisplayProduct.query.all()
    return render_template('fetch.html', data=products)

@app.route('/product/<string:pid>')
def product_detail(pid):
    product_info = None
    pname = None

    # 1. Check in displayproduct (pid is integer)
    if pid.isdigit():
        query_display = text("SELECT * FROM displayproduct WHERE pid = :pid")
        result_display = db.session.execute(query_display, {"pid": int(pid)}).fetchone()

        if result_display:
            product_info = {
                'ID': result_display[0],   # pid
                'Name': result_display[1], # pname
                'ReviewCount': result_display[2],
                'Brand': result_display[3],
                'ImageURL': result_display[4],
                'Rating': result_display[5],
                'Description': result_display[6] if len(result_display) > 6 else "No description available.",
                'Price': result_display[8] if len(result_display) > 8 else None,
            }
            pname = product_info['Name']

    # 2. If not found, check in products (pid = productId string)
    if not product_info:
        query_product = text("SELECT * FROM products WHERE productId = :pid")
        result_product = db.session.execute(query_product, {"pid": pid}).fetchone()

        if result_product:
            product_info = {
                'ID': result_product[1],   # productId
                'Name': result_product[2], # productname
                'ReviewCount': result_product[3],
                'Brand': result_product[4],
                'ImageURL': result_product[5],
                'Rating': result_product[6],
                'Description': result_product[7] if len(result_product) > 7 else "No description available.",
                'Price': result_product[9] if len(result_product) > 11 else None,
            }
            pname = product_info['Name']

    # 3. If still not found, fallback to CSV dataset
    if not product_info:
        result_csv = train_data[train_data['ID'] == (int(pid) if pid.isdigit() else pid)]
        if not result_csv.empty:
            row = result_csv.iloc[0]
            product_info = {
                'ID': row['ID'],
                'Name': row['Name'],
                'ReviewCount': row['ReviewCount'],
                'Brand': row['Brand'],
                'ImageURL': row['ImageURL'],
                'Rating': row['Rating'],
                'Description': row.get('Description', ''),
                'Price': row.get('Price', None),
            }
            pname = row['Name']
        else:
            return "Product not found", 404

    # 4. Recommendations
    recommendations1 = content_based_recommendations(train_data, pname, top_n=5)
    message = None
    if pname not in train_data['Name'].values:
        message = f"No recommendations available for the product '{pname}' as it is not found in the dataset."

    return render_template(
        'product_detail.html',
        product=product_info,
        recommendations1=recommendations1.to_dict(orient='records'),
        message=message
    )

    



@app.route("/signup", methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        repassword = request.form['repassword']

        user_q = text("Select username from signup where username=:username")
        result = db.session.execute(user_q, {'username': username}).scalar()

        if result:
            products = DisplayProduct.query.all()
            return render_template('index.html', data=products, message="Username already exists, try again")

        if password == repassword and len(password) > 8:
            new_signup = Signup(username=username, email=email, password=password, repassword=repassword, status=1,
                                role='user')
            db.session.add(new_signup)
            db.session.commit()
            signup_message = "User signed up successfully"
        elif len(password) < 8:
            signup_message = "Password length should be greater than 8"
        else:
            signup_message = "Unable to Join"
        products = DisplayProduct.query.all()
        return render_template('index.html', data=products, message=signup_message)

    products = DisplayProduct.query.all()
    return render_template('index.html', data=products, message="Please sign up")


@app.route("/signin", methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']

        user = Signup.query.filter_by(username=username, password=password, status=1, role="user").first()
        admins = Admin.query.filter_by(adminName=username, adminPassword=password, role="admin").first()

        if user:
            session['userid'] = user.id
            session['username'] = user.username
            session['logged_in'] = True
            signin_message = "Welcome"
            products = DisplayProduct.query.all()
            return render_template('index.html', data=products, message=signin_message, value=user.username)
        elif admins:
            session['adminlogin'] = admins.id
            session['adminlogin'] = True
            signin_message = "Welcome Admin"
            return render_template('./admin/admin.html', signin_message=signin_message)
        elif not user:
            signin_message = "Invalid Username"
        elif user and user.password != password:
            signin_message = "Invalid Password"
        else:
            signin_message = "You are no longer able to login"

        products = DisplayProduct.query.all()
        return render_template('index.html', data=products, message=signin_message)
    products = DisplayProduct.query.all()
    return render_template('index.html', data=products, message="Please log in")


@app.route("/logout")
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    session.pop('userid', None)
    return redirect(url_for('index'))


@app.route("/search", methods=['GET'])
def search():
    q = request.args.get('query')
    if q:
        results = Products.query.filter(Products.productname.ilike(f"%{q}%")).all()
        recommendations1 = content_based_recommendations(train_data, q, top_n=10)
        message = None
        if recommendations1.empty:
            message = f"No recommendations found for '{q}'. Item not in the training data."
        return render_template('search.html', results=results, recommendations1=recommendations1.to_dict(orient='records'),
                               message=message)
    else:
        return "No search query provided.", 400


@app.route("/cart", methods=['POST', 'GET'])
def cart():
    if 'logged_in' not in session:
        return redirect(url_for('signin'))
    else:
        user = session['userid']

    pid = request.form.get("pid")
    pname = request.form.get("pname")
    price = request.form.get("price")
    image = request.form.get("image")

    if pid and pname and price and image:
        check = text("Select * from carts where userid = :userid and productid = :productid ")
        checkresult = db.session.execute(check, {'userid': user, 'productid': pid}).fetchone()

        if checkresult:
            updata = text("Update carts set quantity = quantity+1 where userid = :userid and productid = :productid")
            db.session.execute(updata, {'userid': user, 'productid': pid})
            db.session.commit()
        else:
            adddata = text("""
                INSERT INTO carts (userid, productid, productname, quantity, image, price) 
                VALUES (:userid, :productid, :productname, 1, :image, :price)
            """)
            db.session.execute(adddata,
                               {'userid': user, 'productid': pid, 'productname': pname, 'image': image, 'price': price})
            db.session.commit()

    query = text("Select * from carts where userid =  :userid")
    result = db.session.execute(query, {'userid': user}).fetchall()

    cart_items = []
    if result:
        for item in result:
            cart_items.append({
                'pid': item[2],
                'pname': item[3],
                'quantity': item[4],
                'image': item[5],
                'price': item[6]
            })

    return render_template('cart.html', cartdata=cart_items)


@app.route('/checkout', methods=['POST', 'GET'])
def checkout():
    pid = request.form.get('pid')
    pname = request.form.get('pname')
    price = request.form.get('price')
    quantity = request.form.get('quantity')
    image = request.form.get('image')
    user = session.get('userid')

    # update cart quantity
    query = text("Update carts set quantity=:quantity where userid=:userid and productid=:productid  ")
    db.session.execute(query, {'quantity': quantity, 'userid': user, 'productid': pid})
    db.session.commit()

    totalprice = int(quantity) * float(price)

    delquery = text("Delete from carts where userid=:userid and productid=:productid")
    db.session.execute(delquery, {'userid': user, 'productid': pid})
    db.session.commit()

    insertquery = text("""
        INSERT INTO purchase (productid, productname, quantity, productprice, userid)
        VALUES (:productid, :productname, :quantity, :productprice, :userid)
    """)
    db.session.execute(insertquery,
                       {'productid': pid, 'productname': pname, 'quantity': quantity, 'productprice': totalprice,
                        'userid': user})
    db.session.commit()

    esewa_info = {
        'amount': totalprice,
        'tax_amount': 0,
        'total_amount': totalprice,
        'transaction_uuid': f"{user}-{pid}-{uuid.uuid4()}",
        'product_code': 'EPAYTEST',
        'product_service_charge': 0,
        'product_delivery_charge': 0,
        'success_url': url_for('esewa_success', _external=True),
        'failure_url': url_for('esewa_failure', _external=True),
        'signed_field_names': 'total_amount,transaction_uuid,product_code',
        'pid': pid,
        'pname': pname,
        'price': totalprice,
        'quantity': quantity,
        'image': image
    }

    session['esewa_info'] = esewa_info
    secret_key = "8gBm/:&EnhH.1/q"
    data_to_sign = f"total_amount={esewa_info['total_amount']},transaction_uuid={esewa_info['transaction_uuid']},product_code={esewa_info['product_code']}"
    esewa_info['signature'] = gen_sha256(secret_key, data_to_sign)

    return render_template('checkout.html', esewa_info=esewa_info)


@app.route('/esewa_success', methods=['GET', 'POST'])
def esewa_success():
    esewa_info = session.pop('esewa_info', None)
    return render_template('success.html', message="Payment Successful. Thank you for your purchase!", esewa_info=esewa_info)


@app.route('/esewa_failure', methods=['GET', 'POST'])
def esewa_failure():
    if 'logged_in' not in session:
        return redirect(url_for('signin'))
    else:
        user = session['userid']
    pid = request.form.get("pid")
    pname = request.form.get("pname")
    price = request.form.get("price")
    image = request.form.get("image")

    if pid and pname and price and image:
        check = text("Select * from carts where userid = :userid and productid = :productid ")
        checkresult = db.session.execute(check, {'userid': user, 'productid': pid}).fetchone()

        if checkresult:
            updata = text("Update carts set quantity = quantity+1 where userid = :userid and productid = :productid")
            db.session.execute(updata, {'userid': user, 'productid': pid})
            db.session.commit()
        else:
            adddata = text("""
              INSERT INTO carts (userid, productid, productname, quantity, image, price) 
              VALUES (:userid, :productid, :productname, 1, :image, :price)
          """)
            db.session.execute(adddata,
                               {'userid': user, 'productid': pid, 'productname': pname, 'image': image, 'price': price})
            db.session.commit()

    query = text("Select * from carts where userid =  :userid")
    result = db.session.execute(query, {'userid': user}).fetchall()

    cart_items = []
    if result:
        for item in result:
            cart_items.append({
                'pid': item[2],
                'pname': item[3],
                'quantity': item[4],
                'image': item[5],
                'price': item[6]
            })

    return render_template('cart.html', cartdata=cart_items)


@app.route('/removeItem', methods=['POST'])
def removeitem():
    pid = request.form.get("pid")
    user = session['userid']
    query = text("delete from carts where userid =:userid and productid=:productid ")
    db.session.execute(query, {'userid': user, 'productid': pid})
    db.session.commit()
    return redirect(url_for('cart'))


@app.route('/detail')
def detail():
    if 'logged_in' not in session:
        return render_template('signin.html')

    user = session['userid']
    query = text("select * from purchase where userid=:userid")
    result = db.session.execute(query, {'userid': user}).fetchall()
    db.session.commit()
    return render_template('detail.html', result=result)


@app.route("/admin")
def admin():
    selectcount = text("select count(purchaseid) from purchase ")
    count = db.session.execute(selectcount).fetchone()

    newpurchase = text("select * from purchase order by purchaseid DESC limit 2")
    purchasecount = db.session.execute(newpurchase).fetchall()

    newuser = text("select * from signup order by id DESC limit 2")
    usercount = db.session.execute(newuser).fetchall()

    selectproduct = text("select count(ID) from products ")
    prodcount = db.session.execute(selectproduct).scalar()

    selectproduct1 = text("select count(pid) from displayproduct ")
    prodcount1 = db.session.execute(selectproduct1).scalar()

    value = (prodcount1 or 0) + (prodcount or 0)

    price = text("select sum(productprice) from purchase")
    pricecount = db.session.execute(price).scalar()

    selectuser = text("select count(id) from signup ")
    user = db.session.execute(selectuser).fetchone()

    return render_template('./admin/admin.html', totalcount=count, user=user, product=value, price=pricecount,
                           newuser=usercount, newpurchase=purchasecount)


@app.route("/adminusers")
def adminusers():
    query = text("Select * from signup where status =1")
    user = db.session.execute(query).fetchall()
    db.session.commit()
    return render_template('./admin/adminusers.html', users=user)


@app.route("/adminlogout")
def adminlogout():
    session.pop('adminlogin', None)
    return redirect(url_for('index'))


@app.route("/adminusersdeactive")
def adminusersdeactive():
    query = text("Select * from signup where status =0")
    user = db.session.execute(query).fetchall()
    db.session.commit()
    return render_template('admin/adminusersdeactivate.html', users=user)


@app.route("/activateuser", methods=['POST'])
def activateuser():
    userid = request.form.get('userid')
    username = request.form.get('username')
    query = text("update signup set status = 1 where id=:userid and username=:username")
    db.session.execute(query, {'userid': userid, 'username': username})
    db.session.commit()
    return render_template('admin/activateuser.html')


@app.route("/removeuser", methods=['POST', 'GET'])
def removeuser():
    userid = request.form.get('userid')
    username = request.form.get('username')
    query = text("update signup set status = 0 where id=:userid and username=:username")
    db.session.execute(query, {'userid': userid, 'username': username})
    db.session.commit()
    return render_template('admin/removeuser.html')


@app.route("/products")
def products():
    query = text("Select * from displayproduct")
    result = db.session.execute(query).fetchall()
    db.session.commit()

    query2 = text("Select * from products")
    result2 = db.session.execute(query2).fetchall()
    db.session.commit()

    return render_template('admin/products.html', result=result, result2=result2)


@app.route("/editproduct", methods=['POST'])
def editproduct():
    productid = request.form.get("productid")
    productname = request.form.get("productname")

    query = text("Select * from displayproduct where pid=:productid and pname=:productname")
    result = db.session.execute(query, {'productid': productid, 'productname': productname})
    db.session.commit()

    query1 = text("Select * from products where productId=:pid and productname=:pname")
    result2 = db.session.execute(query1, {'pid': productid, 'pname': productname})
    db.session.commit()

    return render_template('admin/editproduct.html', result=result, datas=result2)


@app.route("/changedata", methods=['POST'])
def changedata():
    pid = request.form.get('id')
    pname = request.form.get('pname')
    reviewcount = request.form.get('reviewcount')
    brand = request.form.get('brand')
    imageurl = request.form.get('imageurl')
    rating = request.form.get('rating')
    description = request.form.get('description')
    category = request.form.get('category')
    price = request.form.get('price')

    query = text("Select * from displayproduct where pid=:pid and pname=:pname")
    result = db.session.execute(query, {'pid': pid, 'pname': pname})
    db.session.commit()

    query1 = text("Select * from products where productId=:pid and productname=:pname")
    result1 = db.session.execute(query1, {'pid': pid, 'pname': pname})
    db.session.commit()

    if result:
        upquery = text(
            "Update displayproduct set pname=:pname, reviewcount=:reviewcount, brand=:brand, imageurl=:imageurl,"
            "rating=:rating, description=:description, category=:category, price=:price where pid=:pid")
        db.session.execute(upquery, {
            'pid': pid,
            'pname': pname,
            'reviewcount': reviewcount,
            'brand': brand,
            'imageurl': imageurl,
            'rating': rating,
            'description': description,
            'category': category,
            'price': price
        })
        db.session.commit()

    if result1:
        upquery2 = text(
            "Update products set productname=:pname, reviewcount=:reviewcount, productbrand=:brand, imageurl=:imageurl,"
            "rating=:rating, description=:description, category=:category, price=:price where productId=:productId ")
        db.session.execute(upquery2, {
            'productId': pid, 'pname': pname, 'reviewcount': reviewcount, 'brand': brand,
            'imageurl': imageurl,
            'rating': rating, 'description': description, 'category': category,
            'price': price
        })
        db.session.commit()

    return render_template('admin/changedata.html')


@app.route("/delete", methods=['POST'])
def delete():
    id_ = request.form.get('productid')
    name = request.form.get('productname')

    if id_ and id_.strip().isdigit():
        query = text("Delete from displayproduct where pid=:id and pname=:name")
        db.session.execute(query, {'id': id_, 'name': name})
        db.session.commit()
    else:
        query1 = text("Delete from products where productId=:id and productname=:name")
        db.session.execute(query1, {'id': id_, 'name': name})
        db.session.commit()

    return render_template('admin/deleteproduct.html')


@app.route("/purchase")
def purchase():
    query = text("Select * from purchase")
    result = db.session.execute(query).fetchall()

    namelist = []
    for name in result:
        # name is a row; userid is at index 6 in your earlier schema (purchaseid, productid, productname, quantity, productprice, purchaseTime, userid)
        try:
            userid = name[6]
            namelist.append(userid)
        except Exception:
            pass

    nameresult = []
    if namelist:
        namequery = text("Select id, username from signup where id in :id")
        nameresult = db.session.execute(namequery, {'id': tuple(namelist)}).fetchall()
        db.session.commit()

    return render_template('admin/purchase.html', results=result, names=nameresult)


@app.route("/addproduct")
def addproduct():
    return render_template('admin/addproduct.html')


@app.route("/insert", methods=['POST'])
def insert():
    id_ = request.form.get('id', '')
    name = request.form.get('name', '')
    reviewcount = request.form.get('reviewcount', '0')
    brand = request.form.get('brand', '')
    imageurl = request.form.get('imageurl', '')
    rating = request.form.get('rating', '0')
    description = request.form.get('description', '')
    category = request.form.get('category', '')
    price = request.form.get('price', '0')

    # basic empty check
    if not id_.strip() or not name.strip():
        message = "Value cannot be empty"
        return render_template('admin/addproduct.html', message=message)

    if id_.isdigit():
        query = text("""
                    INSERT INTO displayproduct 
                    (pname, reviewcount, brand, imageurl, rating, description, category, price)
                    VALUES (:name, :reviewcount, :brand, :imageurl, :rating, :description, :category, :price)
                    """)
        db.session.execute(query, {'name': name, 'reviewcount': reviewcount, 'brand': brand, 'imageurl': imageurl,
                                   'rating': rating, 'description': description, 'category': category, 'price': price})
        db.session.commit()
    else:
        query = text("""
                            INSERT INTO products 
                            (productId,productname, reviewcount, productbrand, imageurl, rating, description, category, price)
                            VALUES (:id,:name, :reviewcount, :brand, :imageurl, :rating, :description, :category, :price)
                            """)
        db.session.execute(query, {'id': id_, 'name': name, 'reviewcount': reviewcount, 'brand': brand,
                                   'imageurl': imageurl, 'rating': rating, 'description': description,
                                   'category': category, 'price': price})
        db.session.commit()

    return render_template('admin/insert.html')


@app.route("/category")
def category():
    query = text("Select * from category")
    result = db.session.execute(query).fetchall()
    return render_template('admin/category.html', result=result)


@app.route("/deletecategory", methods=['POST'])
def deletecategory():
    pname = request.form.get('productname')
    pid = request.form.get('productid')
    query = text("Delete from category where id=:pid and Categories=:pname")
    db.session.execute(query, {'pid': pid, 'pname': pname})
    db.session.commit()
    return render_template('admin/deletecategory.html')


@app.route("/addcategory")
def addcategory():
    return render_template('admin/addcategory.html')


@app.route("/insertcategory", methods=['POST'])
def insertcategory():
    cname = request.form.get('name')
    query = text("insert into category (Categories) values (:name)")
    db.session.execute(query, {'name': cname})
    db.session.commit()
    return render_template('admin/insertcategory.html')


@app.route("/heads")
def heads():
    return render_template('heads.html')


@app.route("/foots")
def foots():
    return render_template('foots.html')


@app.route("/categories")
def categories_view():
    return render_template('categories.html')


@app.route("/showinfo", methods=['POST'])
def showinfo():
    category = request.form.get('category')

    if category == 'all':
        query = text("select * from products ")
        result = db.session.execute(query)
    else:
        query = text("SELECT * FROM products WHERE category LIKE :category;")
        result = db.session.execute(query, {'category': f'%{category}%'})

    return render_template('categories.html', products1=result)


@app.route("/categorybutton")
def categorybutton():
    return render_template('categorybutton.html')


@app.route("/about")
def about():
    return render_template('about.html')


@app.route("/services") 
def services():
    return render_template('services.html')


# ----------------------
# Start app
# ----------------------
if __name__ == '__main__':
    # Create DB tables if not present (optional)
    try:
        db.create_all()
    except Exception:
        # If you prefer not to auto-create, you can remove create_all()
        pass

    app.run(debug=True)
