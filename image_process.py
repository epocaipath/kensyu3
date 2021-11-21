import cv2
import numpy as np
import openslide
import glob
import os
from PIL import Image
import sys


def Rinkaku(input_dir, floor):
    out_dir = input_dir + "\\futidori"
    # 出力フォルダが無ければ作成
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # 入力フォルダ内のファイルを取得
    os.chdir(input_dir)
    files = glob.glob("*.jpg")
    for file in files:
        image = cv2.imread(file)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_adth = cv2.adaptiveThreshold(img_gray, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,43,3)   #2直化
        img_adth_er = cv2.erode(img_adth, None, iterations = 7)
        img_adth_er = cv2.dilate(img_adth_er, None, iterations = 5) #ノイズ処理
        img_adth_er_re = cv2.bitwise_not(img_adth_er) #白黒反転
        contours, _ = cv2.findContours(img_adth_er_re,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE) #輪郭を抽出
        # 二値化画像(img_adth_er_re)と同じ形状で、0を要素とするndarrayをマスク画像として用意する。
        # (二値化画像と同じ大きさの黒い画像)
        # データ型はopencv用にnp.uint8にする。
        img_mask_adth = np.zeros_like(img_adth, dtype=np.uint8)
        for contour in contours:
        # floorより大きい面積のcontourが対象
            if cv2.contourArea(contour) > floor:
            # 黒色のマスク画像に白色で輪郭を描き、内部を塗りつぶす
            # 引数1=入力画像(要素0のimg_mask)、
    #         引数2=輪郭のリスト
    #         引数3=描きたい輪郭のindex(-1➡全輪郭を描く)
    #         引数4=色(255➡白)
    #         引数5=線の太さ(-1➡塗りつぶし)
                cv2.drawContours(img_mask_adth, [contour],-1,255,-1)
        # 黒い部分を確実な背景部とするために、演習⓹でできた画像を膨張させて白い円の領域を大きくする
        sure_bg_adth = cv2.dilate(img_mask_adth,None,iterations=2)
        # 各組織部(白色円)の中心部分からの距離をcv2.distanceTransformで求める
        # distanceTransform(入力画像,距離関数の種類(cv2.DIST_L2=ユークリッド距離),距離変換に用いるマスクの大きさ)
        dist_transform_adth = cv2.distanceTransform(img_mask_adth,cv2.DIST_L2,5)
    # 距離画像(dist_transform_adth)から、確実な前景画像を作成
        # 各組織部の中心からの距離を閾値にして二値化する。
        # 中心部は255で、離れるにつれて色が暗くなるので、今回は最大値255×0.2 = 51を閾値とする。
        ret, sure_fg_adth = cv2.threshold(dist_transform_adth,
                                        0.2*dist_transform_adth.max(),
                                        255, 0)
        # 「確実な背景部」(sure_bg_adth)から「確実な前景部」(sure_fg_adth)を引くことで、unknown領域が得られる。
        sure_fg_adth = np.uint8(sure_fg_adth)
        # cv2.subtractでsure_bg_adthからsure_fg_adthを引く
        unknown_adth = cv2.subtract(sure_bg_adth, sure_fg_adth)
        # 前景の1オブジェクトごとにラベル（番号）を振っていく
        # cv2.connectedComponents() 関数は画像の背景部に0というラベルを与え，それ以外の物体(各前景部)に対して1から順にラベルをつけていく処理をする。．
        ret, markers_adth = cv2.connectedComponents(sure_fg_adth)
        markers_2_adth = markers_adth+1
        # unknownの領域を0にする。
        # unknown_adth画像で白色の部分をmrkers_2_adthの0にする
        markers_2_adth[unknown_adth==255] = 0
        # 次元を合わせるため、演習⓸の画像(img_mask_adth)を3チャンネルにする (元は輪郭を描いただけなので、1チャンネル)
        img_mstack_adth = np.dstack([img_mask_adth]*3)
        # watershedアルゴリズムに適応する
        markers_water_adth = cv2.watershed(img_mstack_adth, markers_2_adth)
        # 境界の領域(-1)を赤で塗る
        image[markers_water_adth == -1] = [0,0,255]
        # ファイル名は元のndpiファイル名の拡張子を.jpgに変更したものにする。
        file_name = file.split(".")[0]+".jpg"
        # ファイルパスを入力フォルダから出力フォルダに変換する。
        file_name = file_name.replace(f"{input_dir}\\", f"{out_dir}\\")
        # 保存する。
        x = cv2.imwrite(os.path.join(out_dir, file_name), image)
    return x



def Henkan(input_dir, bai):
    out_dir = input_dir  + "\\henkan"
    # 出力フォルダが無ければ作成
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # 入力フォルダ内のファイルを取得
    os.chdir(input_dir)
    files = glob.glob("*.ndpi")
    for file in files:
        # ndpiファイルを開く
        image = openslide.OpenSlide(file)
        # イメージで読み込む(指定したレベルで読み込む)、右下の座標を指定するときに解像度に合わせて
        rgba_img = image.read_region((0,0), bai,(int(image.dimensions[0]/image.level_downsamples[bai]),int(image.dimensions[1]/image.level_downsamples[bai])))
        # イメージをrgbaからrgbへ変換
        rgb_img = rgba_img.convert("RGB")
        # ファイル名は元のndpiファイル名の拡張子を.jpgに変更したものにする。
        file_name = file.split(".")[0]+".jpg"
        # ファイルパスを入力フォルダから出力フォルダに変換する。
        file_name = file_name.replace(f"{input_dir}\\", f"{out_dir}\\")
        # 保存する。
        x = rgb_img.save(os.path.join(out_dir, file_name), "JPEG")
    return x


# フォルダ分け用の関数(img_cutから呼び出し)
def get_tumor_pixnum(img,img_x,x_csize,y_csize,x_start,y_start):
    # 赤色ピクセルの変数の初期化(腫瘍部分)
    t = 0
    # img内(つまりim2=リサイズして切ったアノテーション画像)の色のデータを返す。P形式なら1次元、RGB形式ならタプル。
    # 今回はP形式なので1次元で0か1か2が入っているデータが、画像の左上端から右方向へ順に入っている。
    # つまり、imgdataにはアノテーション画像の全てのピクセル(xy座標)の色のデータが入ることになる。
    imgdata = img.getdata()
    for i in range(y_start,y_start + y_csize):
        # 処理速度を速めるため、forの外で出来る計算は出来るだけ外でしておく。
        # pixdataの算出に必要な画像の横幅と縦の計算は、for j in range～の外で出来るのでここで先に行っておき、
        # for j in range～の中でのpixdataの計算に定数として用いる。
        y_cache = i * img_x
        for j in range(x_start,x_start + x_csize):
            # imgdataのタプルの中の、y_cacheは高さi=0の時は左から順にj番目を見ていく。j=0なら左上端、j=1なら一番上の左から2番目
            # 高さi=1の時は、y_cache=xの幅 + jとすることで2段目の左からのピクセルを示している。
            pixdata = imgdata[y_cache + j]
            # P=1(赤色) → t = 1(腫瘍)
            # P=1で赤色の場合だけtを1ずつ増やしていく。
            if pixdata == 1:
                t += 1
    return t,x_start,y_start

# 画像切り取り用の関数(切る幅は引数で設定)


def img_cut(input_dir_w, input_dir_a, x_csize, y_csize, num_of_img, res_level, sep):
# 入力wsiに対して入力アノテーションpngに追加しているファイル名の接尾辞
    input_at = ".png"
# 出力ファイルの接尾辞
    out_dir_suf = "TUMOR_CUT"
    # 入力wsiフォルダ内のndpiファイルを取得
    os.chdir(input_dir_w)
    files_w = glob.glob('*.ndpi')
    # フォルダからファイルを読み込む
    for file_w in files_w:
        # 画像毎に赤色ピクセル数と、画像の左上端x,y座標を保管するリスト
        # 初期化する
        t_number = []
        #追加部分
        # ndpiファイルを開く
        wsi_image = openslide.OpenSlide(file_w)
        # イメージで読み込む(指定したレベルで読み込む)
        im = wsi_image.read_region((0,0),res_level, (int(wsi_image.dimensions[0]/wsi_image.level_downsamples[res_level]),int(wsi_image.dimensions[1]/wsi_image.level_downsamples[res_level])))
        # デフォルトのリサイズできる最大サイズはセーフティネットとして178956970pixelで設定されている。
        # 今回アノテーションpngファイルをimのサイズにリサイズできるように変更する。
        # ※res_level=0~4を想定,res_level=5の場合は逆に以下の式で最大値が小さくなるので要注意
        Image.MAX_IMAGE_PIXELS = im.size[0]*im.size[1]

        # アノテーション画像(png)を読み込む
        # wsiフォルダパスを用いてアノテーションフォルダパスに変更する
        file_at = input_dir_a + sep + os.path.splitext(os.path.basename(file_w))[0]+input_at
        im2 = Image.open(file_at)

        # im画像サイズ
        # 余白の端は切り捨て
        img_x = im.size[0] // x_csize * x_csize
        img_y = im.size[1] // y_csize * y_csize
        # アノテーション画像(png)を
        # 元イメージimのサイズに変更する。
        im2 = im2.resize((im.size[0], im.size[1]))
        # im2画像そのもの
        # imで画像の余白を切り捨てるので、
        # im2も画像の余白の端を同じ形で切り捨てる必要がある。
        # .cropの範囲指定は(左,上,右,下)の順
        im2 = im2.crop((0,0,img_x,img_y))
        #追加部分ここまで
        # 画像サイズのyを切り取りサイズで割った回数繰り返す。(例: 画像のy=32を8ずつ切りたい場合=> 32/8 = 4回)
        for i in range(int(img_y / y_csize)):
            # 画像サイズのxを切り取りサイズで割った回数繰り返す。(例: 画像のx=24を8ずつ切りたい場合=> 24/8 = 3回)
            for j in range(int(img_x / x_csize)):
                # get_tumor_pixnum関数を呼び出して、切り取り範囲内の赤色の数を取得する。
                t,x_tl,y_tl=get_tumor_pixnum(im2,img_x,x_csize,y_csize,x_csize*j,y_csize*i)
                # 赤のピクセル数が0よりも多い場合だけt_numberリストに保存
                if t > 0:
                    # 赤の数tと、該当画像の左上端のx座標とy座標をリストにしてt_numberリストに追加する。
                    t_number.append([t,[x_tl,y_tl]])
        # sortは破壊的メソッドなので元の配列が上書きされる。
        # デフォルトで昇順のため、reverse = Trueとすることで腫瘍部分のピクセル数を降順にしている。
        # ※腫瘍部のピクセル数が同じ場合は、図の右下部から優先して拾われる。
        t_number.sort(reverse = True)
        for xy in t_number[:num_of_img]:
            img_c = im.crop((xy[1][0],xy[1][1],xy[1][0] + x_csize,xy[1][1] + y_csize))
            out_dir = input_dir_w + sep + out_dir_suf
            # 接尾辞だけでなく出力フォルダ名全体を指定する場合。replaceは完全一致する文字列を置き換える。linuxは("/")
            # 出力フォルダが無ければ作成
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            # imageファイルなのでsaveメソッドで保存する。
            # 入力wsiファイルのフォルダ名部分を取り除く。
            # さらに、spitext[0]でファイル名の拡張子以外の部分だけを取り出す
            file_wt = os.path.splitext(os.path.basename(file_w))[0]
            img_c.save(os.path.join(out_dir, f"{file_wt}_tumor[{xy[1][0]}x{xy[1][1]}].png"))
            # jpgファイルで保存する場合は、以下のようにrgba->rgbに変換してから保存する。


def count_white_area(img, x_csize, y_csize):
    #グレースケールに変換
    img_grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #ぼかし
    kernel = np.ones((10, 10), np.float32)/100
    blur_img = cv2.filter2D(img_grayimg, -1, kernel)

    #二値化
    #画像サイズにより適宜調整を！！
    bi_img = cv2.adaptiveThreshold(
        blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 7)

    #近傍の定義
    neiborhood = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)

    # 縮小、膨張
    img_dilate = cv2.dilate(bi_img, neiborhood, iterations=3)
    img_erode = cv2.erode(img_dilate, neiborhood, iterations=20)

    #白（背景）のピクセル数を算出
    white = cv2.countNonZero(img_erode)
    white_area = white / (x_csize*y_csize)

    return white_area


def img_cut_normal(input_dir_w, x_csize, y_csize, res_level, num_of_img, sep):
    # 出力ファイルの接尾辞
    out_dir_suf = "NORMAL_CUT"
    out_dir = str(input_dir_w + sep + out_dir_suf)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # 入力wsiフォルダ内のndpiファイルを取得
    os.chdir(input_dir_w)
    files_w = glob.glob('*.ndpi')
    # フォルダからファイルを読み込む
    for file_w in files_w:
        t_number = []
        wsi_normal = openslide.OpenSlide(file_w)
        im = wsi_normal.read_region((0,0), res_level, (int(wsi_normal.dimensions[0]/wsi_normal.level_downsamples[res_level]),int(wsi_normal.dimensions[1]/wsi_normal.level_downsamples[res_level])))
        new_img = np.array(im, dtype=np.uint8)
        cv2_img = cv2.cvtColor(new_img, cv2.COLOR_RGBA2BGRA)
        # ※res_level=0~4を想定,res_level=5の場合は逆に以下の式で最大値が小さくなるので要注意
        Image.MAX_IMAGE_PIXELS = im.size[0]*im.size[1]
        # im画像サイズ
        # 余白の端は切り捨て
        img_x = im.size[0] // x_csize * x_csize
        img_y = im.size[1] // y_csize * y_csize
        #追加部分ここまで
        # 画像サイズのyを切り取りサイズで割った回数繰り返す。(例: 画像のy=32を8ずつ切りたい場合=> 32/8 = 4回)
        for i in range(int(img_y / y_csize)):
            # 画像サイズのxを切り取りサイズで割った回数繰り返す。(例: 画像のx=24を8ずつ切りたい場合=> 24/8 = 3回)
            for j in range(int(img_x / x_csize)):
                img1 = cv2_img[y_csize * i: y_csize *
                               (i+1), x_csize * j: x_csize * (j+1)]
                t = round((100-count_white_area(img1, x_csize, y_csize)*100), 0)
                if t > 1:
                    t_number.append([t, img1])
        t_number = sorted(t_number, key=lambda x: (x[0]), reverse=True)
        i=0
        while i < num_of_img:
            img_c = t_number[i][1]
            cv2.imwrite(out_dir+ f"\\{i}_normal[{t_number[i][0]}%].jpg", img_c)
            i += 1
