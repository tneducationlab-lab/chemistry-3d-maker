import streamlit as st
import numpy as np
import trimesh
from ase import Atoms
from ase.build import bulk, molecule
from ase.neighborlist import neighbor_list
from ase.data import vdw_radii, atomic_numbers
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

def create_manual_graphite():
    a = 2.461; c = 6.708; b_ortho = a * np.sqrt(3)
    positions = [[0.0, 0.0, 0.0], [a/2, b_ortho/6, 0.0], [0.0, b_ortho/3, c/2], [a/2, b_ortho/2, c/2]]
    atoms = Atoms(symbols='C4', positions=positions, cell=[a, b_ortho, c], pbc=True)
    atoms = atoms.repeat((2, 2, 2)); atoms.center()
    return atoms

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

def create_carbon_mesh(atoms, style, scale, atom_r_scale, bond_r, cut_cell, show_cell_frame, is_crystal):
    target_cell = atoms.get_cell().diagonal() * scale if is_crystal else None
    
    if is_crystal:
        exp_atoms = atoms.repeat((2, 2, 2)) if len(atoms) < 20 else atoms 
        positions = exp_atoms.get_positions() * scale; symbols = exp_atoms.get_chemical_symbols()
        calc_atoms = exp_atoms
    else:
        positions = atoms.get_positions() * scale; symbols = atoms.get_chemical_symbols()
        calc_atoms = atoms
        
    meshes = []
    
    # Atoms
    for pos, symbol in zip(positions, symbols):
        if is_crystal:
            margin = 0.1 * scale
            if not (-margin<=pos[0]<=target_cell[0]+margin and -margin<=pos[1]<=target_cell[1]+margin and -margin<=pos[2]<=target_cell[2]+margin): continue
        anum = atomic_numbers.get(symbol, 6); base_r = vdw_radii[anum] if vdw_radii[anum] else 1.5
        r = base_r * scale * atom_r_scale if style=="Space Filling (充填)" else 0.4*scale*atom_r_scale
        subdiv = 4 if (cut_cell and style=="Space Filling (充填)" and is_crystal) else 3
        sphere = trimesh.creation.icosphere(subdivisions=subdiv, radius=r); sphere.apply_translation(pos)
        
        if style=="Space Filling (充填)" and cut_cell and is_crystal:
            trimmed = trim_mesh_to_box(sphere, target_cell)
            if trimmed: meshes.append(trimmed)
        else: meshes.append(sphere)

    # Bonds
    if style != "Space Filling (充填)":
        cutoff = 1.8 # C-C bonds
        i_l, j_l, d_l = neighbor_list('ijd', calc_atoms, cutoff=cutoff)
        bond_set = set(); [bond_set.add((i, j)) for i, j in zip(i_l, j_l) if i < j]
        for i, j in bond_set:
            p1=positions[i]; p2=positions[j]; mid=(p1+p2)/2
            if is_crystal:
                m = 0.8*scale
                if not (-m<=mid[0]<=target_cell[0]+m and -m<=mid[1]<=target_cell[1]+m and -m<=mid[2]<=target_cell[2]+m): continue
            vec=p2-p1; ln=np.linalg.norm(vec)
            if ln>1e-6:
                cyl=trimesh.creation.cylinder(radius=bond_r*scale, height=ln, sections=10); ax=np.cross([0,0,1],vec)
                rot=trimesh.transformations.rotation_matrix(np.arccos(np.dot([0,0,1],vec)/ln),ax) if np.linalg.norm(ax)>1e-6 else np.eye(4)
                cyl.apply_transform(trimesh.transformations.translation_matrix((p1+p2)/2) @ rot)
                if cut_cell and is_crystal:
                    trimmed = trim_mesh_to_box(cyl, target_cell)
                    if trimmed: meshes.append(trimmed)
                else: meshes.append(cyl)

    if not meshes: return None
    combined = trimesh.util.concatenate(meshes)
    if show_cell_frame and is_crystal:
        frame = create_unit_cell_frame(target_cell, scale)
        if frame: combined = trimesh.util.concatenate([combined, frame])
    try: combined.fix_normals()
    except: pass
    return combined

st.set_page_config(page_title="炭素の同素体メーカー", page_icon="💎", layout="wide")
st.title("💎 炭素の同素体メーカー")

sel = st.sidebar.selectbox("物質を選ぶ", ["Diamond (ダイヤモンド)", "Graphite (黒鉛)", "Fullerene (フラーレン C60)"])
is_crystal = False
if sel == "Diamond (ダイヤモンド)": atoms = bulk('C', 'diamond', a=3.567, cubic=True); is_crystal = True
elif sel == "Graphite (黒鉛)": atoms = create_manual_graphite(); is_crystal = True
elif sel == "Fullerene (フラーレン C60)": atoms = molecule('C60'); atoms.center()

st.sidebar.header("2. モデル設定")
style = st.sidebar.selectbox("スタイル", ["Ball and Stick (球棒)", "Space Filling (充填)"])
scale = st.sidebar.slider("サイズ倍率", 5.0, 15.0, 10.0)
frame = False; cut = False; atom_s = 1.0; bond_r = 0.0

if style == "Ball and Stick (球棒)":
    bond_r = st.sidebar.slider("棒の太さ", 0.05, 0.30, 0.15)
    if is_crystal:
        frame = st.sidebar.checkbox("単位格子の枠を表示", value=True)
        cut = st.sidebar.checkbox("枠からはみ出た結合をカット", value=True)
else:
    atom_s = st.sidebar.slider("原子の重なり", 0.9, 1.5, 1.1)
    if is_crystal:
        frame = st.sidebar.checkbox("単位格子の枠を表示", value=False)
        cut = st.sidebar.checkbox("単位格子で切断 (教科書風)", value=True)

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader(sel)
    try: fig, ax = plt.subplots(); ap=atoms.copy(); ap.rotate(15,'x'); ap.rotate(45,'y'); plot_atoms(ap, ax, radii=0.4, rotation=('0x,0y,0z')); ax.set_axis_off(); st.pyplot(fig)
    except: pass
with c2:
    if st.button("モデル作成 (OBJ形式)", type="primary"):
        with st.spinner("計算中..."):
            mesh = create_carbon_mesh(atoms, style, scale, atom_s, bond_r, cut, frame, is_crystal)
            if mesh:
                p = "carbon.obj"; mesh.export(p, file_type='obj')
                with open(p, "r") as f: d = f.read()
                st.success("完了！"); st.download_button("OBJダウンロード", d, "carbon.obj", "text/plain")