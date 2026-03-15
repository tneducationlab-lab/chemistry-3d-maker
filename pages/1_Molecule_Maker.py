import streamlit as st
import numpy as np
import trimesh
import pubchempy as pcp
from ase import Atoms
from ase.build import molecule
from ase.neighborlist import neighbor_list
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

def create_molecule_mesh(atoms, style, scale, atom_r_scale, bond_r):
    positions = atoms.get_positions() * scale
    symbols = atoms.get_chemical_symbols()
    meshes = []
    for pos, symbol in zip(positions, symbols):
        anum = atomic_numbers.get(symbol, 6)
        base_r = vdw_radii[anum] if vdw_radii[anum] else 1.5
        r = base_r * scale * atom_r_scale if style=="Space Filling (充填)" else (0.25 if symbol=='H' else 0.4)*scale*atom_r_scale
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=r)
        sphere.apply_translation(pos); meshes.append(sphere)
    
    if style != "Space Filling (充填)":
        cutoff = 2.0
        i_l, j_l, d_l = neighbor_list('ijd', atoms, cutoff=cutoff)
        bond_set = set(); [bond_set.add((i, j)) for i, j in zip(i_l, j_l) if i < j]
        for i, j in bond_set:
            p1=positions[i]; p2=positions[j]; vec=p2-p1; ln=np.linalg.norm(vec)
            if ln>1e-6:
                cyl=trimesh.creation.cylinder(radius=bond_r*scale, height=ln, sections=10)
                z=np.array([0,0,1]); ax=np.cross(z,vec)
                rot=trimesh.transformations.rotation_matrix(np.arccos(np.dot(z,vec)/ln), ax) if np.linalg.norm(ax)>1e-6 else np.eye(4)
                cyl.apply_transform(trimesh.transformations.translation_matrix((p1+p2)/2) @ rot); meshes.append(cyl)
    if not meshes: return None
    combined = trimesh.util.concatenate(meshes)
    try: combined.fix_normals()
    except: pass
    return combined

st.set_page_config(page_title="分子模型メーカー", page_icon="🧬", layout="wide")
st.title("🧬 分子模型メーカー")
mode = st.sidebar.radio("検索モード", ["代表的な分子", "キーワード検索"])
atoms=None; name_display=""
PRESET_DATA = {"Water": "Water (水 H2O)", "Carbon dioxide": "Carbon dioxide (CO2)", "Ammonia": "Ammonia (NH3)", "Methane": "Methane (CH4)", "Ethanol": "Ethanol (エタノール)", "Benzene": "Benzene (ベンゼン)"}

if mode == "代表的な分子":
    sel = st.sidebar.selectbox("物質名", list(PRESET_DATA.values()))
    tgt = [k for k, v in PRESET_DATA.items() if v == sel][0]
    try: atoms=molecule(tgt); atoms.center(); name_display=tgt
    except: pass
elif mode == "キーワード検索":
    inp = st.sidebar.text_input("物質名を入力"); q = translate_input(inp)
    if q:
        try:
            cids = pcp.get_cids(q, 'name', record_type='3d')
            if cids:
                compound = pcp.Compound.from_cid(cids[0], record_type='3d')
                symbols = [a.element for a in compound.atoms]; positions = [(a.x, a.y, a.z) for a in compound.atoms]
                atoms = Atoms(symbols=symbols, positions=positions); atoms.center(); name_display=f"{inp} ({q})"
        except: pass

style = st.sidebar.selectbox("スタイル", ["Ball and Stick (球棒)", "Space Filling (充填)"])
scale = st.sidebar.slider("サイズ倍率", 5.0, 15.0, 10.0)
atom_s = 1.0; bond_r = 0.0
if style == "Ball and Stick (球棒)": bond_r = st.sidebar.slider("棒の太さ", 0.05, 0.30, 0.15)
else: atom_s = st.sidebar.slider("原子の重なり", 0.9, 1.5, 1.1)

if atoms:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader(name_display)
        try: fig, ax = plt.subplots(); ap=atoms.copy(); ap.rotate(15,'x'); ap.rotate(45,'y'); plot_atoms(ap, ax, radii=0.4, rotation=('0x,0y,0z')); ax.set_axis_off(); st.pyplot(fig)
        except: pass
    with c2:
        if st.button("モデル作成 (OBJ形式)", type="primary"):
            with st.spinner("計算中..."):
                mesh = create_molecule_mesh(atoms, style, scale, atom_s, bond_r)
                if mesh:
                    p = "molecule.obj"; mesh.export(p, file_type='obj')
                    with open(p, "r") as f: d = f.read()
                    st.success("完了！"); st.download_button("OBJダウンロード", d, f"{name_display}.obj", "text/plain")