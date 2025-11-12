"""生成済み DICOM を用いてマニュアル用イメージを作成します。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from app import (IMG_DIR, compute_default_window, extract_plane,
                 load_dicom_series, window_level_image)

OUTPUT_DIR = Path(__file__).resolve().parent / 'docs' / 'images'

plt.rcParams['font.family'] = ['Yu Gothic', 'MS Gothic', 'Meiryo', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def format_title(prefix: str, ww: float, wl: float) -> str:
    return f"{prefix}\nWW: {int(round(ww))} / WL: {int(round(wl))}"


def create_overview(volume) -> None:
    z_dim, y_dim, x_dim = volume.shape
    default_ww, default_wl = compute_default_window(volume)
    axial_idx = z_dim // 2
    sagittal_idx = x_dim // 2

    axial = window_level_image(extract_plane(volume, 'Axial', axial_idx), default_ww, default_wl)
    sagittal = window_level_image(extract_plane(volume, 'Sagittal', sagittal_idx), default_ww, default_wl)

    fig = plt.figure(figsize=(12, 6))
    gridspec = fig.add_gridspec(2, 2, height_ratios=[4, 1.5])
    ax_axial = fig.add_subplot(gridspec[0, 0])
    ax_reform = fig.add_subplot(gridspec[0, 1])
    ax_footer = fig.add_subplot(gridspec[1, :])

    ax_axial.imshow(axial, cmap='gray')
    ax_axial.axvline(sagittal_idx, color='#66b3ff', linewidth=2)
    ax_axial.set_title('① Axial（元画像）', loc='left', fontsize=12, fontweight='bold')
    ax_axial.set_axis_off()
    ax_axial.text(10, 30, '③ 再構成位置ガイド線', color='white', fontsize=11,
                  bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.5))

    ax_reform.imshow(sagittal, cmap='gray')
    ax_reform.set_title('② Sagittal（再構成）', loc='left', fontsize=12, fontweight='bold')
    ax_reform.set_axis_off()

    ax_footer.set_axis_off()
    ax_footer.set_xlim(0, 10)
    ax_footer.set_ylim(0, 1)

    # Tab-like UI
    # Tab bar
    ax_footer.add_patch(patches.Rectangle((0.2, 0.75), 9.6, 0.15, color='#e0e0e0', linewidth=1.2, edgecolor='#4d4d4d'))
    # Active Tab (Navigation)
    ax_footer.add_patch(patches.Rectangle((0.3, 0.75), 2.0, 0.15, color='#f5f5f5', linewidth=1.2, edgecolor='#4d4d4d'))
    ax_footer.text(0.6, 0.8, 'ナビゲーション', fontsize=9, fontweight='bold')
    ax_footer.text(2.6, 0.8, 'Windowing', fontsize=9)
    ax_footer.text(4.4, 0.8, '情報', fontsize=9)

    # Main content area for the tab
    ax_footer.add_patch(patches.FancyBboxPatch((0.2, 0.08), 9.6, 0.67,
                                               boxstyle='round,pad=0.02',
                                               linewidth=1.2, edgecolor='#4d4d4d', facecolor='#f5f5f5'))
    ax_footer.text(0.45, 0.4, '④ 操作タブ（ナビゲーション、Windowing、情報）', fontsize=11)
    ax_footer.text(0.45, 0.2, '⑤ 終了ボタン', fontsize=11)

    fig.suptitle('アプリ全体の構成', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    fig.savefig(OUTPUT_DIR / 'overview.png', dpi=220)
    plt.close(fig)


def create_windowing_example(volume) -> None:
    z_dim, _, _ = volume.shape
    axial_idx = z_dim // 2
    default_ww, default_wl = compute_default_window(volume)

    soft_ww = default_ww
    soft_wl = default_wl
    bone_ww = max(default_ww * 0.45, 250)
    bone_wl = default_wl + default_ww * 0.6

    axial_soft = window_level_image(extract_plane(volume, 'Axial', axial_idx), soft_ww, soft_wl)
    axial_bone = window_level_image(extract_plane(volume, 'Axial', axial_idx), bone_ww, bone_wl)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].imshow(axial_soft, cmap='gray')
    axes[0].set_title(format_title('標準ウィンドウ', soft_ww, soft_wl), fontsize=11)
    axes[0].set_axis_off()

    axes[1].imshow(axial_bone, cmap='gray')
    axes[1].set_title(format_title('骨ウィンドウ例', bone_ww, bone_wl), fontsize=11)
    axes[1].set_axis_off()

    fig.suptitle('Window Width / Window Level 調整例', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_DIR / 'windowing.png', dpi=220)
    plt.close(fig)


def create_plane_switch_example(volume) -> None:
    _, y_dim, x_dim = volume.shape
    default_ww, default_wl = compute_default_window(volume)
    sagittal_idx = x_dim // 2
    coronal_idx = y_dim // 2

    sagittal = window_level_image(extract_plane(volume, 'Sagittal', sagittal_idx), default_ww, default_wl)
    coronal = window_level_image(extract_plane(volume, 'Coronal', coronal_idx), default_ww, default_wl)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].imshow(sagittal, cmap='gray')
    axes[0].set_title('Sagittal（矢状断）', fontsize=12)
    axes[0].set_axis_off()
    axes[0].annotate('左右方向の位置関係を確認', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    axes[1].imshow(coronal, cmap='gray')
    axes[1].set_title('Coronal（冠状断）', fontsize=12)
    axes[1].set_axis_off()
    axes[1].annotate('前後方向の広がりを確認', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    fig.suptitle('Plane 切り替えによる断面表示', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_DIR / 'plane_switch.png', dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_output_dir()
    volume, _ = load_dicom_series(str(IMG_DIR))
    create_overview(volume)
    create_windowing_example(volume)
    create_plane_switch_example(volume)
    print('Saved manual assets to', OUTPUT_DIR)


if __name__ == '__main__':
    main()
