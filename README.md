# 🧪 3D化学模型メーカー (3D Chemistry Model Maker)

中学・高校の化学で学ぶ主要な分子や結晶構造の、3Dプリント用データ（OBJファイル）を作成するWebアプリケーションです。
1つのアプリ内でモードを切り替えることで、用途に合わせた多様なモデルを生成できます。直感的な立体構造の理解や、授業での活用に役立ちます。

## ✨ 主な機能 (Features)

本アプリは、左側のサイドバーから以下のモードを切り替えて使用します。

### 1. 🧬 分子模型メーカー (Molecule Maker)
水、メタン、フラーレンといった代表的な単一分子のモデルを作成します。
* **キーワード検索対応:** 日本語または英語で物質名を入力すると、PubChemのデータベースから構造を自動取得して立体化します。
* **スタイルの変更:** 「球棒モデル（Ball and Stick）」と「充填モデル（Space Filling）」を切り替えられます。

### 2. 🧊 単位格子メーカー (Unit Cell Maker)
鉄(BCC)、銅(FCC)、塩化ナトリウムなどの結晶モデルを作成します。黒鉛（グラファイト）の層状構造にも対応しています。
* **教科書完全準拠のカット機能:** 最大の特徴は、教科書の図と同じように**「単位格子の枠でスパッと切断された状態（面の1/2原子や、角の1/8原子）」**を正確に再現できる点です。枠からはみ出た余計な結合の棒も綺麗にカットされます。

## 🖨️ 3Dプリント時の造形のコツ (Printing Tips)

ダウンロードしたOBJファイルは、お使いのスライサーソフト（Bambu Studio, Cura, PrusaSlicerなど）でスライス処理をしてから造形してください。

**💡 重要：推奨設定 (Recommended Settings)**
球体を含むモデルを綺麗に出力するため、スライサー上では以下の設定を推奨します。
1.  オブジェクトをビルドプレート（床面）から少し（数mm程度）浮かせる。
2.  「サポート材を生成する（Generate Support）」をオンにし、モデルの下部全体をサポートで支えるようにする。

---

# 🇺🇸 English Version

This web application generates 3D printable data (OBJ format) for major molecules and crystal unit cells studied in junior high and high school chemistry. 

## ✨ Features

You can switch between the following modes from the left sidebar:

### 1. 🧬 Molecule Maker
Generate models of single molecules like water, methane, and fullerene.
* **Keyword Search:** Search for any molecule by name (powered by PubChem) to generate its 3D structure.
* **Custom Styles:** Switch between "Ball and Stick" and "Space Filling" models.

### 2. 🧊 Unit Cell Maker
Generate crystal models such as Iron (BCC), Copper (FCC), and NaCl. It also supports the layered structure of Graphite.
* **Textbook-Accurate Slicing:** The core feature is its ability to perfectly slice atoms and bonds exactly at the unit cell boundaries (e.g., 1/2 atoms on faces, 1/8 atoms at corners), reproducing standard textbook illustrations.

## 🖨️ 3D Printing Tips

Please process the downloaded OBJ files through your slicer software (e.g., Bambu Studio, Cura) before printing.

**💡 Recommended Settings**
To ensure high-quality prints of the spherical models, we highly recommend:
1.  Lifting the object slightly off the build plate in your slicer.
2.  Enabling support structures so that the entire bottom of the model is properly supported.

---
**Built with:** Python, [Streamlit](https://streamlit.io/), [Trimesh](https://trimsh.org/), and [ASE (Atomic Simulation Environment)](https://wiki.fysik.dtu.dk/ase/).
