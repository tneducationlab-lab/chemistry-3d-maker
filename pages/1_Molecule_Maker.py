import streamlit as st
import numpy as np
import trimesh
import pubchempy as pcp
from ase import Atoms
from ase.data import vdw_radii, atomic_numbers
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from deep_translator import GoogleTranslator

def translate_input(text):
    if not text: return text
    try:
        if any(ord(char) > 128 for char in text):
            return GoogleTranslator(source='auto', target='en').translate(text)
    except: pass
    return text

def fetch_molecule_data(cid=None, name=None):
    try:
        if name:
            cids = pcp.get_cids(name, 'name', record_type='3d')
            if not cids: return None, None, None
            cid = cids[0]
            
        c3d = pcp.Compound.from_cid(cid, record_type='3d')
        bonds = c3d.bonds
        # 3Dデータに結合情報がない場合は、2Dデータから正確な結合情報を引っ張ってくる
        if not bonds:
            c2d = pcp.Compound.from_cid(cid)
            bonds = c2d.bonds
            
        atoms_data = {a.aid: {'symbol': a.element, 'pos': np.array([a.x, a.y, a.z])} for a in c3d.atoms}
        # 結合する原子のIDと、結合の多重度(1, 2, 3)を取得
        bonds_data = [(b.aid1, b.aid2, b.order) for b in bonds]
        
        symbols = [a.element for a in c3d.atoms]
        positions = [(a.x, a.y, a.z) for a in c3d.atoms]
        ast = Atoms(symbols=symbols, positions=positions)
        ast.center()
        
        # 万が一データベースに結合情報が全く無い場合のみ、距離で単結合を推測（安全装置）
        if not bonds_data:
            from ase.neighborlist import neighbor_list
            i_l, j_l, d_l = neighbor_list('ijd', ast, cutoff=2.0)
            bond_set = set()
            for i, j in zip(i_l, j_l):
                if i < j: bond_set.add((i, j))
            aid_list = list(atoms_data.keys())
            bonds_data = [(aid_list[i], aid_list[j], 1) for i, j in bond_set]

        return atoms_data, bonds_data, ast
    except Exception as e:
        return None, None, None

def create_advanced_molecule_mesh(atoms_data, bonds_data, style, scale, atom_r_scale, bond_r):
    meshes = []
    is_space_filling = (style == "Space Filling (充填)")
    
    # --- 原子の配置 ---
    for aid, adata in atoms_data.items():
        pos = adata['pos'] * scale
        symbol = adata['symbol']
        anum = atomic_numbers.get(symbol, 6)
        base_r = vdw_radii[anum] if (anum < len(vdw_radii) and vdw_radii[anum] is not None) else 1.5
        
        r = base_r * scale * atom_r_scale if is_space_filling else (0.25 if symbol=='H' else 0.4) * scale * atom_r_scale
        sphere = trimesh.creation.icosphere(subdivisions=4 if is_space_filling else 3, radius=r)
        
        rot_hack = trimesh.transformations.rotation_matrix(0.123, [1, 1, 1])
        sphere.apply_transform(rot_hack)
        sphere.apply_translation(pos)
        meshes.append(sphere)
        
    # --- 結合棒の配置 ---
    if not is_space_filling:
        # 結合の向きを平面に揃えるため、各原子の隣接関係を把握
        neighbors = {aid: [] for aid in atoms_data.keys()}
        for a1, a2, _ in bonds_data:
            if a1 in neighbors and a2 in neighbors:
                neighbors[a1].append(a2)
                neighbors[a2].append(a1)
        
        # 二重結合などの「棒と棒の隙間」の広さ
        offset_dist = bond_r * scale * 1.4 
        
        for aid1, aid2, order in bonds_data:
            if aid1 not in atoms_data or aid2 not in atoms_data: continue
            p1 = atoms_data[aid1]['pos'] * scale
            p2 = atoms_data[aid2]['pos'] * scale
            
            vec = p2 - p1
            ln = np.linalg.norm(vec)
            if ln < 1e-6: continue
            
            # 周囲の原子から「分子の平面」を計算し、棒を並べる美しい方向を決定
            offset_dir = None
            for n_aid in neighbors[aid1]:
                if n_aid != aid2:
                    pn = atoms_data[n_aid]['pos'] * scale
                    normal = np.cross(vec, pn - p1)
                    if np.linalg.norm(normal) > 1e-3:
                        offset_dir = np.cross(vec, normal)
                        offset_dir /= np.linalg.norm(offset_dir)
                        break
            if offset_dir is None:
                ref = np.array([1.0, 0.0, 0.0])
                if np.abs(np.dot(vec/ln, ref)) > 0.9: ref = np.array([0.0, 1.0, 0.0])
                offset_dir = np.cross(vec, ref)
                offset_dir /= np.linalg.norm(offset_dir)
            
            # 結合次数(1, 2, 3)の判定
            o = int(order) if isinstance(order, (int, float)) else 1
            if o > 3 or o <= 0: o = 1
            
            offsets = []
            if o == 1:
                offsets = [np.array([0.0, 0.0, 0.0])]
            elif o == 2:
                offsets = [offset_dir * offset_dist, -offset_dir * offset_dist]
            elif o == 3:
                offsets = [np.array([0.0, 0.0, 0.0]), offset_dir * offset_dist * 1.8, -offset_dir * offset_dist * 1.8]
            
            # 計算した本数と位置に棒を配置
            for off in offsets:
                cyl = trimesh.creation.cylinder(radius=bond_r*scale, height=ln, sections=10)
                z_axis = np.array([0,0,1])
                ax = np.cross(z_axis, vec)
                if np.linalg.norm(ax) < 1e-6:
                    rot = np.eye(4) if vec[2] > 0 else trimesh.transformations.rotation_matrix(np.pi, [1,0,0])
                else:
                    ang = np.arccos(np.dot(z_axis, vec)/ln)
                    rot = trimesh.transformations.rotation_matrix(ang, ax)
                
                mid = (p1 + p2)/2 + off
                cyl.apply_transform(trimesh.transformations.translation_matrix(mid) @ rot)
                meshes.append(cyl)
                
    if not meshes: return None
    combined = trimesh.util.concatenate(meshes)
    try: combined.fix_normals()
    except: pass
    return combined

st.set_page_config(page_title="分子模型メーカー", page_icon="🧬", layout="wide")
st.title("🧬 分子模型メーカー")
mode = st.sidebar.radio("検索モード", ["代表的な分子", "キーワード検索"])

atoms_data = None; bonds_data = None; ase_atoms = None; name_display = ""

# すぐに呼び出せるプリセット（メニューにも二重/三重結合がわかるものを追加しました）
PRESETS = {
    "Water (水 H2O)": 962,
    "Carbon dioxide (CO2)": 280,
    "Ammonia (NH3)": 222,
    "Methane (CH4)": 297,
    "Ethanol (エタノール)": 702,
    "Benzene (ベンゼン)": 241,
    "Ethylene (エチレン / 二重結合)": 6325,
    "Acetylene (アセチレン / 三重結合)": 6326
}

if mode == "代表的な分子":
    sel = st.sidebar.selectbox("物質名", list(PRESETS.keys()))
    atoms_data, bonds_data, ase_atoms = fetch_molecule_data(cid=PRESETS[sel])
    if atoms_data: name_display = sel
    else: st.error("データの取得に失敗しました。")

elif mode == "キーワード検索":
    inp = st.sidebar.text_input("物質名を入力"); q = translate_input(inp)
    if q:
        atoms_data, bonds_data, ase_atoms = fetch_molecule_data(name=q)
        if atoms_data: name_display = f"{inp} ({q})"
        else: st.error("データが見つかりませんでした。")

style = st.sidebar.selectbox("スタイル", ["Ball and Stick (球棒)", "Space Filling (充填)"])
scale = st.sidebar.slider("サイズ倍率", 5.0, 15.0, 10.0)
atom_s = 1.0; bond_r = 0.0

if style == "Ball and Stick (球棒)": 
    bond_r = st.sidebar.slider("結合棒の太さ", 0.05, 0.12, 0.01, step=0.01)
else: 
    atom_s = st.sidebar.slider("原子の重なり", 0.9, 1.5, 1.1)

if atoms_data:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader(name_display)
        try: 
            fig, ax = plt.subplots()
            ap=ase_atoms.copy()
            ap.rotate(15,'x'); ap.rotate(45,'y')
            
            if style == "Space Filling (充填)":
                rad_list = []
                for s in ap.get_chemical_symbols():
                    anum = atomic_numbers.get(s, 6)
                    base_r = vdw_radii[anum] if (anum < len(vdw_radii) and vdw_radii[anum] is not None) else 1.5
                    rad_list.append(base_r * atom_s * 0.5)
                plot_atoms(ap, ax, radii=np.array(rad_list), rotation=('0x,0y,0z'))
            else:
                plot_atoms(ap, ax, radii=0.4, rotation=('0x,0y,0z'))
                
            ax.set_axis_off()
            st.pyplot(fig)
        except: pass
    with c2:
        if st.button("モデル作成 (OBJ形式)", type="primary"):
            with st.spinner("計算中..."):
                mesh = create_advanced_molecule_mesh(atoms_data, bonds_data, style, scale, atom_s, bond_r)
                if mesh and not mesh.is_empty:
                    p = "molecule.obj"
                    mesh.export(p)
                    with open(p, "rb") as f:
                        st.download_button("📥 OBJダウンロード", f, file_name="molecule.obj")
                    st.success("完了！二重結合・三重結合も完璧に反映されました。")
                else:
                    st.error("メッシュの生成に失敗しました。")
else:
    st.info("👈 左のメニューから物質を選択するか、検索してください")
