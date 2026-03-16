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
    return atoms

def create_lattice_frame(width, height, depth, thickness=0.2):
    lines = [
        ([0,0,0], [width,0,0]), ([0,0,0], [0,height,0]), ([0,0,0], [0,0,depth]),
        ([width,height,depth], [0,height,depth]), ([width,height,depth], [width,0,depth]), ([width,height,depth], [width,height,0]),
        ([width,0,0], [width,height,0]), ([width,0,0], [width,0,depth]),
        ([0,height,0], [width,height,0]), ([0,height,0], [0,height,depth]),
        ([0,0,depth], [width,0,depth]), ([0,0,depth], [0,height,depth])
    ]
    frame_meshes = []
    for start, end in lines:
        p1 = np.array(start); p2 = np.array(end)
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length < 1e-6: continue

        cyl = trimesh.creation.cylinder(radius=thickness, height=length, sections=8)
        z = np.array([0, 0, 1])
        ax = np.cross(z, vec)
        if np.linalg.norm(ax) < 1e-6:
            rot = np.eye(4) if vec[2] > 0 else trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        else:
            ang = np.arccos(np.dot(z, vec) / length)
            rot = trimesh.transformations.rotation_matrix(ang, ax)

        cyl.apply_transform(trimesh.transformations.translation_matrix((p1 + p2) / 2) @ rot)
        frame_meshes.append(cyl)
    return trimesh.util.concatenate(frame_meshes) if frame_meshes else None

def safe_slice(mesh, normal, origin):
    if mesh is None or mesh.is_empty: return None
    bounds = mesh.bounds
    tol = 1e-4
    if normal[0] == 1:
        if bounds[0][0] >= origin[0] - tol: return mesh
        if bounds[1][0] <= origin[0] + tol: return None
    elif normal[0] == -1:
        if bounds[1][0] <= origin[0] + tol: return mesh
        if bounds[0][0] >= origin[0] - tol: return None
    elif normal[1] == 1:
        if bounds[0][1] >= origin[1] - tol: return mesh
        if bounds[1][1] <= origin[1] + tol: return None
    elif normal[1] == -1:
        if bounds[1][1] <= origin[1] + tol: return mesh
        if bounds[0][1] >= origin[1] - tol: return None
    elif normal[2] == 1:
        if bounds[0][2] >= origin[2] - tol: return mesh
        if bounds[1][2] <= origin[2] + tol: return None
    elif normal[2] == -1:
        if bounds[1][2] <= origin[2] + tol: return mesh
        if bounds[0][2] >= origin[2] - tol: return None
    return trimesh.intersections.slice_mesh_plane(mesh, plane_normal=normal, plane_origin=origin, cap=True)

def create_carbon_mesh(atoms, style, scale, atom_s, bond_thickness_ratio, cut_cell, show_cell_frame, is_crystal):
    target_cell = atoms.get_cell().diagonal() * scale if is_crystal else None
    positions = atoms.get_positions() * scale
    symbols = atoms.get_chemical_symbols()
    meshes = []
    
    is_space_filling = (style == "Space Filling (充填)")

    # 原子の生成
    for pos, symbol in zip(positions, symbols):
        if is_crystal:
            margin = 0.1 * scale
            if not (-margin<=pos[0]<=target_cell[0]+margin and -margin<=pos[1]<=target_cell[1]+margin and -margin<=pos[2]<=target_cell[2]+margin): continue
            
        anum = atomic_numbers.get(symbol, 6)
        base_r = vdw_radii[anum] if vdw_radii[anum] else 1.7
        if is_space_filling:
            r = base_r * scale * atom_s * 0.5
        else:
            r = 0.4 * scale * atom_s
            
        subdiv = 4 if (cut_cell and is_space_filling and is_crystal) else 3
        sphere = trimesh.creation.icosphere(subdivisions=subdiv, radius=r)
        
        rot_hack = trimesh.transformations.rotation_matrix(0.123, [1, 1, 1])
        sphere.apply_transform(rot_hack)
        sphere.apply_translation(pos)
        
        if is_space_filling and cut_cell and is_crystal:
            sphere = safe_slice(sphere, [1,0,0], [0,0,0])
            sphere = safe_slice(sphere, [-1,0,0], [target_cell[0],0,0])
            sphere = safe_slice(sphere, [0,1,0], [0,0,0])
            sphere = safe_slice(sphere, [0,-1,0], [0,target_cell[1],0])
            sphere = safe_slice(sphere, [0,0,1], [0,0,0])
            sphere = safe_slice(sphere, [0,0,-1], [0,0,target_cell[2]])
            
        if sphere and not sphere.is_empty:
            meshes.append(sphere)

    # 結合棒の生成
    if not is_space_filling:
        cutoff = 1.8 # 炭素-炭素結合距離の限界値（これ以上遠いものは結合ではない）
        i_l, j_l, d_l = neighbor_list('ijd', atoms, cutoff=cutoff)
        bond_set = set()
        for i, j in zip(i_l, j_l):
            if i < j: bond_set.add((i, j))
            
        bond_radius = scale * bond_thickness_ratio
        max_drawn_length = cutoff * scale # 描画を許可する最大の長さ（スケール後）
        
        for i, j in bond_set:
            p1 = positions[i]; p2 = positions[j]
            vec = p2 - p1
            ln = np.linalg.norm(vec)
            
            # 【重要】周期境界をまたいで端と端をつなぐ「巨大な謎の棒」を除外する安全装置
            if ln > max_drawn_length or ln < 1e-6:
                continue
            
            mid = (p1+p2)/2
            if is_crystal:
                m = 0.8 * scale
                if not (-m<=mid[0]<=target_cell[0]+m and -m<=mid[1]<=target_cell[1]+m and -m<=mid[2]<=target_cell[2]+m): continue
                
            cyl = trimesh.creation.cylinder(radius=bond_radius, height=ln, sections=10)
            ax = np.cross([0,0,1], vec)
            if np.linalg.norm(ax) < 1e-6:
                rot = np.eye(4) if vec[2] > 0 else trimesh.transformations.rotation_matrix(np.pi, [1,0,0])
            else:
                ang = np.arccos(np.dot([0,0,1], vec)/ln)
                rot = trimesh.transformations.rotation_matrix(ang, ax)
            cyl.apply_transform(trimesh.transformations.translation_matrix(mid) @ rot)
            
            if cut_cell and is_crystal:
                cyl = safe_slice(cyl, [1,0,0], [0,0,0])
                cyl = safe_slice(cyl, [-1,0,0], [target_cell[0],0,0])
                cyl = safe_slice(cyl, [0,1,0], [0,0,0])
                cyl = safe_slice(cyl, [0,-1,0], [0,target_cell[1],0])
                cyl = safe_slice(cyl, [0,0,1], [0,0,0])
                cyl = safe_slice(cyl, [0,0,-1], [0,0,target_cell[2]])
                
            if cyl and not cyl.is_empty:
                meshes.append(cyl)

    if not meshes: return None
    combined = trimesh.util.concatenate(meshes)
    
    if show_cell_frame and is_crystal:
        # 枠線は結合棒の「半分の細さ(0.5倍)」にして区別する
        frame_thickness = scale * bond_thickness_ratio * 0.5
        frame = create_lattice_frame(target_cell[0], target_cell[1], target_cell[2], frame_thickness)
        if frame: combined = trimesh.util.concatenate([combined, frame])
        
    try: combined.fix_normals()
    except: pass
    return combined

st.set_page_config(page_title="炭素の同素体メーカー", page_icon="💎", layout="wide")
st.title("💎 炭素の同素体メーカー")

sel = st.sidebar.selectbox("物質を選ぶ", ["Diamond (ダイヤモンド)", "Graphite (黒鉛)", "Fullerene (フラーレン C60)"])
is_crystal = (sel != "Fullerene (フラーレン C60)")

st.sidebar.header("モデル設定")

if is_crystal:
    rep = st.sidebar.slider("繰り返しの数 (XYZ方向)", min_value=1, max_value=5, value=2, help="数を大きくすると壮大な構造になりますが、計算に少し時間がかかります")
else:
    rep = 1

if sel == "Diamond (ダイヤモンド)": 
    atoms = bulk('C', 'diamond', a=3.567, cubic=True)
    if rep > 1: atoms = atoms.repeat((rep, rep, rep))
    atoms.center()
elif sel == "Graphite (黒鉛)": 
    atoms = create_manual_graphite()
    if rep > 1: atoms = atoms.repeat((rep, rep, rep))
    atoms.center()
elif sel == "Fullerene (フラーレン C60)": 
    atoms = molecule('C60')
    atoms.center()

style = st.sidebar.selectbox("スタイル", ["Ball and Stick (球棒)", "Space Filling (充填)"])
scale = st.sidebar.slider("サイズ倍率", 5.0, 15.0, 10.0)
frame = False; cut = False; atom_s = 1.0

# ★スライダーの範囲とデフォルト値を「折れない太さ」に修正しました
if style == "Ball and Stick (球棒)":
    bond_thickness = st.sidebar.slider("結合棒の太さ（※枠線は自動でこの半分の細さになります）", min_value=0.05, max_value=0.30, value=0.12, step=0.01)
    if is_crystal:
        frame = st.sidebar.checkbox("単位格子の外枠を表示", value=True)
        cut = st.sidebar.checkbox("枠からはみ出た結合をカット", value=True)
else:
    bond_thickness = 0.12
    atom_s = st.sidebar.slider("原子の重なり", 0.9, 1.5, 1.1)
    if is_crystal:
        frame = st.sidebar.checkbox("単位格子の枠を表示", value=False)
        cut = st.sidebar.checkbox("単位格子で切断 (教科書風)", value=True)

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader(sel)
    try: 
        fig, ax = plt.subplots()
        ap=atoms.copy()
        ap.rotate(15,'x'); ap.rotate(45,'y')
        plot_atoms(ap, ax, radii=0.4, rotation=('0x,0y,0z'))
        ax.set_axis_off()
        st.pyplot(fig)
    except: pass
with c2:
    if st.button("モデル作成 (OBJ形式)", type="primary"):
        with st.spinner(f"計算中... (繰り返し数 {rep}×{rep}×{rep} は少し時間がかかります)"):
            mesh = create_carbon_mesh(atoms, style, scale, atom_s, bond_thickness, cut, frame, is_crystal)
            if mesh and not mesh.is_empty:
                p = "carbon.obj"
                mesh.export(p)
                with open(p, "rb") as f:
                    st.download_button("📥 OBJダウンロード", f, file_name="carbon_model.obj")
                st.success("完了！")
            else:
                st.error("メッシュの生成に失敗しました。")
