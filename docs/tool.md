# キャリブレーションツールの使い方

チェッカーボード画像からモジュール個体ごとのカメラパラメータを推定し、歪み補正のプレビューまで出すコマンドラインツール。依存は numpy と OpenCV だけで、画面表示は使わない(Jetson に SSH で入って使える)。

動作確認は合成画像でのみ行った。**実機の画像と実カメラからの取り込みは未検証。**

投影モデルとパラメータの定義は `docs/spec.md` を参照。

## 準備

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows は .venv\Scripts\pip
```

## 1. 撮影

- ボードは内側コーナー 9x6、マス 30 mm を既定にしている。違う場合は `--board 横x縦` と `--square マス一辺[m]` で指定する。
- 20〜30 枚。ボード全体が写ること(一部が欠けた画像は検出できない)。
- 画像の中央だけでなく、四隅と外周にもボードを置く。ボードを傾けた画像も混ぜる。
- 全画像を同じ解像度で、1 個体につき 1 フォルダに保存する。

## 2. 推定

```
python -m fovision calibrate --images "shots/M0001/*.png" --module-id M0001 --source real --out results/real
```

- `--module-id`: 個体の識別子。出力ファイル名と JSON に入る。
- `--source`: `real`(実機)か `simulation`。結果を混ぜないために必須にしてある。
- 焦点距離の初期値は不要。候補を 5 通り試して再投影誤差が最小のものを採用する。レンズの公称値を知らなくても動く。
- 再投影誤差が飛び抜けて大きい画像(中央値の 3 倍かつ 1 px 超)は自動で外してやり直す。

実カメラから直接取り込む場合(未検証):

```
python -m fovision calibrate --camera 0 --frames 30 --interval 2 --save-frames --module-id M0001 --source real --out results/real
```

`--camera` にはカメラ番号か GStreamer パイプライン文字列を渡す。`--interval` 秒ごとに 1 枚取り込むので、その間にボードを動かす。

### 出力

| ファイル | 内容 |
|---|---|
| `<個体>.json` | カメラパラメータ(fx, fy, cx, cy, k1〜k4)、再投影誤差、推定標準偏差 |
| `<個体>_report.json` | 使った画像・検出できなかった画像・外した画像、画像ごとの誤差、コーナーの分布、注意事項 |
| `<個体>_coverage.png` | 検出コーナーの分布図。緑が採用、赤が誤差の大きいビュー、黄色の丸が推定した主点 |

### 画面に出る注意の読み方

| 注意 | 対処 |
|---|---|
| 使えた画像が少ない | 撮り足す |
| コーナーの無い区画が多い | 分布図で空いている場所にボードを写して撮り足す |
| 外周が拘束されていない | 画像の端までボードを寄せた画像を足す |
| 像高が単調増加にならない | `--nk 2` などで歪み係数を減らす |
| 再投影誤差が大きい | ボードの寸法指定、ぶれ・ピント、ボードの反りを確認する |

再投影誤差が小さくても、コーナーが届いていない領域の精度は保証されない。分布図とあわせて判断する。

## 3. 歪み補正のプレビュー

```
python -m fovision undistort --params results/real/M0001.json --image shots/M0001/a.png --out a_equirect.png --proj equirect
```

| `--proj` | 特徴 |
|---|---|
| `equirect` | 正距円筒。180° 超の画角もそのまま入る |
| `cylindrical` | 円筒。縦線がまっすぐになる。縦の画角は 120° まで |
| `stereographic` | ステレオ投影。180° 超も入り、局所的な形が保たれる |
| `perspective` | 透視投影。直線がまっすぐになる。160° まで |

`--hfov`、`--vfov`(出力に収める画角)と `--width`(出力幅)で範囲と大きさを変えられる。

## 4. 実機なしで試す

```
python -m fovision simulate  --out _tmp/SIM-0001 --module-id SIM-0001 --perturb
python -m fovision calibrate --images "_tmp/SIM-0001/sim_*.png" --module-id SIM-0001 --source simulation --out _tmp/out
python -m fovision evaluate  --params _tmp/out/SIM-0001.json --truth _tmp/SIM-0001/truth.json
```

`simulate` は合成画像一式と真値(`truth.json`)を作る。`--f` で焦点距離 [px] を変えられ、省略するとイメージサークルが画像幅いっぱいになる値を使う。`evaluate` は推定値と真値の差を出す。

## まだ無いもの

- ボードなしの自己校正(`fovision/selfcal.py`)は点の対応を入力にする段階で、画像からの特徴点追跡が未実装のためツールには入れていない。
- 撮影中に検出状況を画面で見る機能は無い。取り込み後に分布図で確認する。
- 補正用の対応表を保存して動画に連続適用する機能は無い(1 枚ずつの変換のみ)。
