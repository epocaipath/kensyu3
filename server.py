from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import numpy as np
import cv2
from image_process import Rinkaku, Henkan, img_cut, img_cut_normal
from datetime import datetime
import os
import string
import random
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000

############################################フォルダを作成##################################################

SAVE_DIR = "./images"
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

# SAVE_DIR_futidori = "./images/futidori"
# if not os.path.isdir(SAVE_DIR_futidori):
#     os.makedirs(SAVE_DIR_futidori)

# SAVE_DIR_henkan = "./images/henkan"
# if not os.path.isdir(SAVE_DIR_henkan):
#     os.mkdir(SAVE_DIR_henkan)

############################################動かすのに必要なの##################################################


app = Flask(__name__, static_url_path="")
# app.config['UPLOAD_FOLDER'] = SAVE_DIR_henkan


def random_str(n):
    return ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])


############################################CSSを適用させる##################################################
@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

############################################トップページと画像##################################################

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/images/<path:path>')
def send_js(path):
    return send_from_directory(SAVE_DIR, path)


############################################「縁取り」##################################################
@app.route('/futidori')
def futidori():
    return render_template('futidori.html')

@app.route('/finish_futi')
def finish():
    return render_template('finish_futi.html')

@app.route('/upload_futi', methods=['POST'])
def upload_futi():
    if request.method == 'POST':
        file_path = request.form.get('file_path')
        floor = request.form.get('floor', type=int)
        Rinkaku(file_path, floor)
        return redirect('/finish_futi')


############################################「JPG」に変換##################################################
@app.route('/henkan')
def henkan():
    return render_template('henkan.html')

@app.route('/finish_henkan')
def finish_henkan():
    return render_template('finish_henkan.html')

@app.route('/upload_henkan', methods=['POST'])
def upload_henkan():
    if request.method == 'POST':
        file_path = request.form.get('file_path')
        bai = request.form.get('bai', type=int)
        Henkan(input_dir = file_path, bai=bai)
        return redirect('/finish_henkan')

############################################Tumor##################################################

@app.route('/tumor')
def tumor():
    return render_template('tumor.html')

@app.route('/finish_tumor')
def finish_tumor():
    return render_template('finish_tumor.html')

@app.route('/upload_tumor', methods=['POST'])
def upload_tumor():
    if request.method == 'POST':
        file_path_ndpi = request.form.get("file_path_ndpi")
        file_path_png = request.form.get("file_path_png")
        sep = request.form.get("sep")
        size_x = request.form.get('size_x', type=int)
        size_y = request.form.get('size_y', type=int)
        res_level = request.form.get('res_level', type=int)
        num_maisu = request.form.get('num_maisu', type=int)
        img_cut(file_path_ndpi, file_path_png,
                size_x, size_y, num_maisu, res_level, sep)
        return redirect('/finish_tumor')

############################################ normal ##################################################

@app.route('/normal')
def normal():
    return render_template('normal.html')

@app.route('/finish_normal')
def finish_normal():
    return render_template('finish_normal.html')

@app.route('/upload_normal', methods=['POST'])
def upload_normal():
    if request.method == 'POST':
        file_path_ndpi = request.form.get("n_file_path_ndpi")
        nsep = request.form.get("n_sep")
        size_x = request.form.get('n_size_x', type=int)
        size_y = request.form.get('n_size_y', type=int)
        res_level = request.form.get('n_res_level', type=int)
        num_maisu = request.form.get('n_num_maisu', type=int)
        img_cut_normal(input_dir_w = file_path_ndpi,x_csize = size_x,y_csize = size_y, res_level=res_level, num_of_img=num_maisu,sep=nsep)
        return redirect('/finish_normal')

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8888)
