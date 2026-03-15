import streamlit as st
import numpy as np
import trimesh
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import neighbor_list
from ase.data import vdw_radii, atomic_numbers
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

def trim_mesh_to_box(mesh, box_size):
    try:
        if mesh is None or mesh.is_empty: return None
        bounds = mesh.bounds; tol = 1e-4
        if (bounds[0][0]>=-tol and bounds[1][0]<=box_size[0]+tol and bounds[0][1]>=-tol and bounds[1][1]<=box_size[1]+tol and bounds[0][2]>=-tol and bounds[1][2]<=box_size[2]+tol): return mesh
        mesh = trimesh.intersections.slice_mesh_plane(mesh, plane_normal=[1,0,0], plane_origin=[0,0,0], cap=True)
        mesh = trimesh.intersections.slice_mesh_plane(mesh, plane_normal=[-1,0,0], plane_origin=[box_size[0],0,0], cap=True)
        mesh = trimesh.intersections.slice_mesh_plane(mesh, plane_normal=[0,1,0], plane_origin=[0,0,0], cap=True)
        mesh = trimesh.intersections.slice_mesh_plane(mesh, plane_normal=[0,-1,0], plane_origin=[0,box_size[1],0], cap=True)
        mesh = trimesh.intersections.slice_mesh_plane(mesh, plane_normal=[0,0,1], plane_origin=[0,0,0], cap=True)
        mesh = trimesh.intersections.slice_mesh_plane(mesh, plane_normal=[0,0,-1], plane_origin=[0,0,box_size[2]], cap=True)
        return mesh if not mesh.is_empty else None
    except: return None

def create_unit_cell_frame(box_size, scale):
    try:
        x, y, z = box_size; r = 0.015 * scale 
        edges = [([0,0,0],[x,0,0]),([0,0,0],[0,y,0]),([0,0,0],[0,0,z]),([x,0,0],[x,y,0]),([x,0,0],[x,0,z]),([0,y,0],[x,y,0]),([0,y,0],[0,y,z]),([0,0,z],[x,0,z]),([0,0,z],[0,y,z]),([x,y,0],[x,y,z]),([x,0,z],[x,y,z]),([0,y,z],[x,y,z])]
        meshes = []
        for s, e in edges:
            p1=np.array(s); p2=np.array(e); vec=p2-p1; ln=np.linalg.norm(vec)
            if ln>1e-6:
                cyl=trimesh.creation.cylinder(radius=r, height=ln, sections=8); ax=np.cross([0,0,1],vec)
                rot=trimesh.transformations.rotation_matrix(np.arccos(np.dot([0,0,1],vec)/ln),ax) if np.linalg.norm(ax)>1e-6 else np.eye(4)
                cyl.apply_transform(trimesh.transformations.translation_matrix((p1+p2)/2) @ rot); meshes.append(cyl)
        return trimesh.util.concatenate(meshes) if meshes else None
    except: return None

def create_crystal_mesh(atoms, style, scale, atom_r_scale, bond_r, cut_cell, show_cell_frame):
    target_cell = atoms.get_cell().diagonal() * scale 
    exp_atoms = atoms.repeat((2, 2, 2))
    positions = exp_atoms.get_positions() * scale; symbols = exp_atoms.get_chemical_symbols()
    meshes = []
    
    for pos, symbol in zip(positions, symbols):
        margin = 0.1 * scale
        if not (-margin<=pos[0]<=target_cell[0]+margin and -margin<=pos[1]<=target_cell[1]+margin and -margin<=pos[2]<=target_cell[2]+margin): continue
        anum = atomic_numbers.get(symbol, 6); base_r = vdw_radii[anum] if vdw_radii[anum] else 1.5
        r = base_r * scale * atom_r_scale if style=="Space Filling (充填)" else (0.25 if symbol=='H' else 0.4)*scale*atom_r_scale
        subdiv = 4 if (cut_cell and style=="Space Filling (充填)") else 3
        sphere = trimesh.creation.icosphere(subdivisions=subdiv, radius=r); sphere.apply_translation(pos)
        if style=="Space Filling (充填)" and cut_cell:
            trimmed = trim_mesh_to_box(sphere, target_cell)
            if trimmed: meshes.append(trimmed)
        else: meshes.append(sphere)

    if style != "Space Filling (充填)":
        cutoff = 2.9 if "Fe" in symbols or "Cu" in symbols else (3.2 if "Na" in symbols else (4.3 if "Cs" in symbols else 1.7))
        i_l, j_l, d_l = neighbor_list('ijd', exp_atoms, cutoff=cutoff)
        bond_set = set(); [bond_set.add((i, j)) for i, j in zip(i_l, j_l) if i < j]
        for i, j in bond_set:
            p1=positions[i]; p2=positions[j]; mid=(p1+p2)/2; m = 0.8*scale
            if not (-m<=mid[0]<=target_cell[0]+m and -m<=mid[1]<=target_cell[1]+m and -m<=mid[2]<=target_cell[2]+m): continue
            vec=p2-p1; ln=np.linalg.norm(vec)
            if ln>1e-6:
                cyl=trimesh.creation.cylinder(radius=bond_r*scale, height=ln, sections=10); ax=np.cross([0,0,1],vec)
                rot=trimesh.transformations.rotation_matrix(np.arccos(np.dot([0,0,1],vec)/ln),ax) if np.linalg.norm(ax)>1e-6 else np.eye(4)
                cyl.apply_transform(trimesh.transformations.translation_matrix((p1+p2)/2) @ rot)
                if cut_cell:
                    trimmed = trim_mesh_to_box(cyl, target_cell)
                    if trimmed: meshes.append(trimmed)
                else: meshes.append(cyl)

    if not meshes: return None
    combined = trimesh.util.concatenate(meshes)
    if show_cell_frame:
        frame = create_unit_cell_frame(target_cell, scale)
        if frame: combined = trimesh.util.concatenate([combined, frame])
    try: combined.fix_normals()
    except: pass
    return combined

st.set_page_config(page_title="単位格子メーカー", page_icon="🧊", layout="wide")
st.title("🧊 単位格子メーカー (結晶構造)")
st.sidebar.header("1. 物質を選ぶ")
PRESET = {"Iron (鉄/BCC)": ('Fe','bcc',2.866), "Copper (銅/FCC)": ('Cu','fcc',3.615), "Magnesium (マグネシウム/HCP)": ('Mg','hcp',3.21,5.21), "Sodium chloride (NaCl)": ('NaCl','rocksalt',5.64), "Cesium chloride (CsCl)": ('CsCl','cesiumchloride',4.123), "Silicon (ケイ素)": ('Si','diamond',5.43)}
sel = st.sidebar.selectbox("結晶を選択", list(PRESET.keys()))

if sel == "Magnesium (マグネシウム/HCP)": atoms = bulk(PRESET[sel][0], PRESET[sel][1], a=PRESET[sel][2], c=PRESET[sel][3], orthorhombic=True)
else: atoms = bulk(PRESET[sel][0], PRESET[sel][1], a=PRESET[sel][2], cubic=True)

st.sidebar.header("2. モデル設定")
style = st.sidebar.selectbox("スタイル", ["Ball and Stick (球棒)", "Space Filling (充填)"])
scale = st.sidebar.slider("サイズ倍率", 5.0, 15.0, 10.0)
frame = False; cut = True; atom_s = 1.0; bond_r = 0.0
if style == "Ball and Stick (球棒)":
    bond_r = st.sidebar.slider("棒の太さ", 0.05, 0.30, 0.15)
    frame = st.sidebar.checkbox("単位格子の枠を表示", value=True)
    cut = st.sidebar.checkbox("枠からはみ出た結合をカット", value=True)
else:
    atom_s = st.sidebar.slider("原子の重なり", 0.9, 1.5, 1.1)
    frame = st.sidebar.checkbox("単位格子の枠を表示", value=False)
    cut = st.sidebar.checkbox("単位格子で切断 (教科書風)", value=True)

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader(sel)
    try: fig, ax = plt.subplots(); ap=atoms.copy(); ap.rotate(15,'x'); ap.rotate(45,'y'); plot_atoms(ap, ax, radii=0.4, rotation=('0x,0y,0z')); ax.set_axis_off(); st.pyplot(fig)
    except: pass
with c2:
    if st.button("モデル作成 (OBJ形式)", type="primary"):
        with st.spinner("計算中 (カット処理は重いです)..."):
            mesh = create_crystal_mesh(atoms, style, scale, atom_s, bond_r, cut, frame)
            if mesh:
                p = "crystal.obj"; mesh.export(p, file_type='obj')
                with open(p, "r") as f: d = f.read()
                st.success("完了！"); st.download_button("OBJダウンロード", d, "crystal.obj", "text/plain")