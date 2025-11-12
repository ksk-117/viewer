import os
import io
from typing import Dict, Optional, Tuple

import numpy as np
import pydicom
import PySimpleGUI as sg
from PIL import Image, ImageDraw

IMG_DIR = os.path.join(os.path.dirname(__file__), 'img')
PLANE_OPTIONS = ('Sagittal', 'Coronal')

TITLE_FONT = ('Arial', 16, 'bold')
SUBTITLE_FONT = ('Arial', 12)
SECTION_FONT = ('Arial', 13, 'bold')
LABEL_FONT = ('Arial', 11)
CAPTION_FONT = ('Arial', 10)


def load_dicom_series(folder: str) -> Tuple[np.ndarray, list]:
    """Load a single DICOM series from *folder* sorted by slice order."""
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.dcm')]
    if not files:
        raise FileNotFoundError(f'No DICOM files found in {folder}')

    datasets = []
    for file_path in files:
        try:
            ds = pydicom.dcmread(file_path)
            datasets.append(ds)
        except Exception as exc:  # skip unreadable files but keep going
            print('Failed to read', file_path, exc)

    if not datasets:
        raise RuntimeError('No readable DICOM files were found.')

    def sort_key(ds):
        if 'InstanceNumber' in ds:
            return int(ds.InstanceNumber)
        if 'ImagePositionPatient' in ds:
            return float(ds.ImagePositionPatient[2])
        return 0

    datasets.sort(key=sort_key)

    first = datasets[0]
    rows = int(first.Rows)
    cols = int(first.Columns)
    volume = np.zeros((len(datasets), rows, cols), dtype=np.float32)

    for idx, ds in enumerate(datasets):
        pixel_array = ds.pixel_array.astype(np.float32)
        if 'RescaleIntercept' in ds and 'RescaleSlope' in ds:
            pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        volume[idx] = pixel_array

    return volume, datasets


def window_level_image(img: np.ndarray, ww: float, wl: float) -> np.ndarray:
    """Apply window/level to a 2D image and return an 8-bit array."""
    ww = max(float(ww), 1.0)
    low = wl - (ww / 2.0)
    high = wl + (ww / 2.0)
    if high == low:
        return np.zeros_like(img, dtype=np.uint8)
    clipped = np.clip(img, low, high)
    norm = ((clipped - low) / (high - low) * 255.0).astype(np.uint8)
    return norm


def to_pil(img2d: np.ndarray, size: Optional[Tuple[int, int]] = None, overlay_line=None) -> bytes:
    """Convert 2D uint8 array into PNG bytes for PySimpleGUI."""
    im = Image.fromarray(img2d.astype(np.uint8), mode='L')
    if size:
        im = im.resize(size, Image.BILINEAR)
    if overlay_line is not None:
        draw = ImageDraw.Draw(im)
        x1, y1, x2, y2, color = overlay_line
        draw.line((x1, y1, x2, y2), fill=color, width=2)
    bio = io.BytesIO()
    im.save(bio, format='PNG')
    return bio.getvalue()


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(min(int(value), maximum), minimum)


def compute_default_window(volume: np.ndarray) -> Tuple[float, float]:
    low = float(np.percentile(volume, 5))
    high = float(np.percentile(volume, 95))
    ww = max(high - low, 1.0)
    wl = (high + low) / 2.0
    return ww, wl


def extract_plane(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    """Return a 2D slice for the requested plane."""
    z_dim, y_dim, x_dim = volume.shape  # y == rows, x == cols
    if plane == 'Axial':
        index = clamp(index, 0, z_dim - 1)
        return volume[index]
    if plane == 'Sagittal':
        index = clamp(index, 0, x_dim - 1)
        plane_img = volume[:, :, index]
        plane_img = np.rot90(plane_img, k=1)
        return np.flipud(plane_img)
    if plane == 'Coronal':
        index = clamp(index, 0, y_dim - 1)
        plane_img = volume[:, index, :]
        plane_img = np.rot90(plane_img, k=1)
        return np.flipud(plane_img)
    raise ValueError(f'Unknown plane: {plane}')


def plane_limit(volume: np.ndarray, plane: str) -> int:
    _, y_dim, x_dim = volume.shape
    return (x_dim - 1) if plane == 'Sagittal' else (y_dim - 1)


def plane_label(plane: str) -> str:
    return 'Sagittal (矢状断)' if plane == 'Sagittal' else 'Coronal (冠状断)'


def build_metadata_text(datasets: list, volume: np.ndarray) -> str:
    first = datasets[0]
    z_dim, y_dim, x_dim = volume.shape
    spacing = getattr(first, 'PixelSpacing', None)
    slice_thickness = getattr(first, 'SliceThickness', None)

    lines = [
        f'シリーズ枚数: {z_dim}',
        f'画像サイズ: {y_dim} x {x_dim}',
    ]
    if spacing:
        try:
            sx, sy = spacing
            lines.append(f'ピクセル間隔: {float(sx):.2f} mm × {float(sy):.2f} mm')
        except Exception:
            pass
    if slice_thickness:
        try:
            lines.append(f'スライス厚: {float(slice_thickness):.2f} mm')
        except Exception:
            pass
    study = getattr(first, 'StudyDescription', '') or getattr(first, 'SeriesDescription', '')
    if study:
        lines.append(f'シリーズ説明: {study}')
    return '\n'.join(lines)


def main():
    try:
        volume, datasets = load_dicom_series(IMG_DIR)
    except Exception as exc:
        sg.popup_error('Failed to load DICOM series', str(exc))
        return

    z_dim, y_dim, x_dim = volume.shape
    default_ww, default_wl = compute_default_window(volume)
    ww_slider_max = max(int(default_ww * 5), 2000)
    wl_min = int(min(np.floor(np.min(volume)), -1200))
    wl_max = int(max(np.ceil(np.max(volume)), 1600))

    preset_map: Dict[str, Tuple[str, int, int]] = {
        '-PRESET_SOFT-': ('Soft Tissue', int(round(default_ww)), int(round(default_wl))),
        '-PRESET_BONE-': ('Bone', int(round(max(default_ww * 0.45, 250))), int(round(default_wl + default_ww * 0.6))),
        '-PRESET_LUNG-': ('Lung', 1500, -600),
        '-PRESET_HEAD-': ('Head', 120, 40),
    }

    sg.theme('DefaultNoMoreNagging')

    image_size = (520, 520)
    default_plane = PLANE_OPTIONS[0]
    reformat_default = plane_limit(volume, default_plane) // 2

    viewer_left = sg.Column([
        [sg.Text('Axial (元画像)', font=SECTION_FONT)],
        [sg.Image(key='-AXIAL-', size=image_size, background_color='black', pad=(0, 12))],
    ], expand_x=True, expand_y=True, element_justification='center')

    viewer_right = sg.Column([
        [sg.Text('再構成断面', font=SECTION_FONT, key='-REFORM_TITLE-')],
        [sg.Image(key='-REFORM-', size=image_size, background_color='black', pad=(0, 12))],
    ], expand_x=True, expand_y=True, element_justification='center')

    viewer_frame = sg.Frame('', [[viewer_left, sg.VSeparator(), viewer_right]],
                            expand_x=True, expand_y=True, pad=(0, 0), relief=sg.RELIEF_FLAT)

    navigation_frame = sg.Frame('断面ナビゲータ', [
        [sg.Text('表示断面', font=LABEL_FONT),
         sg.Combo(PLANE_OPTIONS, default_value=default_plane, key='-PLANE-', font=LABEL_FONT, readonly=True, enable_events=True, expand_x=True),
         sg.Text('', key='-PLANE_DESC-', size=(18, 1), text_color='#1565C0', font=CAPTION_FONT)],
        [sg.Text('Axial スライス', font=LABEL_FONT)],
        [sg.Slider(range=(0, z_dim - 1), key='-AXIAL_SLICE-', orientation='h', enable_events=True, default_value=z_dim // 2, resolution=1, expand_x=True)],
        [sg.Text('', key='-AXIAL_INFO-', size=(24, 1), font=CAPTION_FONT)],
        [sg.Text('再構成位置', font=LABEL_FONT)],
        [sg.Slider(range=(0, plane_limit(volume, default_plane)), key='-REFORM_SLICE-', orientation='h', enable_events=True, default_value=reformat_default, resolution=1, expand_x=True)],
        [sg.Text('', key='-REFORM_INFO-', size=(24, 1), font=CAPTION_FONT)],
    ], expand_x=True)

    windowing_frame = sg.Frame('Windowing', [
        [sg.Text('Window Width', font=LABEL_FONT), sg.Push(), sg.Text('', key='-WW_INFO-', size=(10, 1), font=CAPTION_FONT)],
        [sg.Slider(range=(1, ww_slider_max), key='-WW-', orientation='h', enable_events=True, default_value=int(round(default_ww)), resolution=1, expand_x=True)],
        [sg.Text('Window Level', font=LABEL_FONT), sg.Push(), sg.Text('', key='-WL_INFO-', size=(10, 1), font=CAPTION_FONT)],
        [sg.Slider(range=(wl_min, wl_max), key='-WL-', orientation='h', enable_events=True, default_value=int(round(default_wl)), resolution=1, expand_x=True)],
        [sg.HorizontalSeparator(pad=((0, 0), (8, 8)))],
        [sg.Column([
            [sg.Button('WW/WL 初期化', key='-RESET-WINDOW-', size=(14, 1), expand_x=True, font=LABEL_FONT),
             sg.Text('', key='-PRESET_INFO-', size=(28, 1), text_color='#2E7D32', font=CAPTION_FONT)],
            [sg.Text('プリセット', font=LABEL_FONT)],
            [sg.Button('Soft Tissue', key='-PRESET_SOFT-', size=(14, 1), expand_x=True, font=LABEL_FONT),
             sg.Button('Bone', key='-PRESET_BONE-', size=(14, 1), expand_x=True, font=LABEL_FONT)],
            [sg.Button('Lung', key='-PRESET_LUNG-', size=(14, 1), expand_x=True, font=LABEL_FONT),
             sg.Button('Head', key='-PRESET_HEAD-', size=(14, 1), expand_x=True, font=LABEL_FONT)],
            [sg.HorizontalSeparator(pad=((0, 0), (8, 8)))],
            [sg.Button('Reset Plane', key='-RESET-PLANE-', size=(14, 1), expand_x=True, font=LABEL_FONT),
             sg.Button('全画面', key='-FULLSCREEN-', size=(14, 1), expand_x=True, font=LABEL_FONT)],
        ], element_justification='center', expand_x=True, pad=(0, 6))],
    ], expand_x=True)

    metadata_frame = sg.Frame('シリーズ情報', [[
        sg.Multiline(build_metadata_text(datasets, volume), key='-METADATA-', size=(60, 10), disabled=True, no_scrollbar=True, font=CAPTION_FONT)
    ]], expand_x=True)

    navigation_tab = sg.Tab('ナビゲーション', [[navigation_frame]], key='-TAB_NAV-', expand_x=True)
    windowing_tab = sg.Tab('Windowing', [[windowing_frame]], key='-TAB_WIN-', expand_x=True)
    metadata_tab = sg.Tab('情報', [[metadata_frame]], key='-TAB_META-', expand_x=True)

    control_column = sg.Column([
        [sg.TabGroup([[navigation_tab, windowing_tab, metadata_tab]], key='-TAB_GROUP-', expand_x=True, expand_y=True)],
        [sg.Push(), sg.Button('終了', key='Quit', size=(12, 1), font=LABEL_FONT)]
    ], expand_x=True, expand_y=True, pad=((12, 0), (0, 0)), element_justification='left')

    layout = [
        [sg.Text('DICOM Explorer', font=TITLE_FONT, text_color='#0D47A1'), sg.Push(), sg.Text('インタラクティブ 3D クロスビューア', font=SUBTITLE_FONT, text_color='#0277BD')],
        [sg.HorizontalSeparator()],
        [sg.Column([[viewer_frame]], expand_x=True, expand_y=True, pad=(0, 0)), sg.VSeparator(), control_column],
        [sg.HorizontalSeparator()],
        [sg.Text('準備完了', key='-STATUS-', size=(120, 1), font=CAPTION_FONT)]
    ]

    window = sg.Window('DICOM Viewer', layout, finalize=True, resizable=True, margins=(12, 12))
    if hasattr(window, 'maximize'):
        window.maximize()
    else:  # fallback for legacy naming
        window.Maximize()

    def parse_values(val_dict: Dict) -> Tuple[str, int, int, int, int, int]:
        plane = val_dict.get('-PLANE-', default_plane)
        axial_idx = clamp(round(val_dict.get('-AXIAL_SLICE-', 0)), 0, z_dim - 1)
        reform_limit = plane_limit(volume, plane)
        reform_idx = clamp(round(val_dict.get('-REFORM_SLICE-', reformat_default)), 0, reform_limit)
        ww = clamp(round(val_dict.get('-WW-', default_ww)), 1, ww_slider_max)
        wl = clamp(round(val_dict.get('-WL-', default_wl)), wl_min, wl_max)
        return plane, axial_idx, reform_idx, ww, wl, reform_limit

    def update_overlay_coords(plane: str, reform_idx: int, reform_limit: int, shape: Tuple[int, int]) -> Tuple[int, int, int, int, int]:
        height, width = shape
        if reform_limit <= 0:
            if plane == 'Sagittal':
                x = width // 2
                return x, 0, x, height - 1, 255
            y = height // 2
            return 0, y, width - 1, y, 255
        ratio = reform_idx / reform_limit if reform_limit else 0
        if plane == 'Sagittal':
            x = int(round(ratio * (width - 1)))
            return x, 0, x, height - 1, 255
        y = int(round(ratio * (height - 1)))
        return 0, y, width - 1, y, 255

    def update_reformat_slider(plane: str, current_idx: int) -> int:
        limit = plane_limit(volume, plane)
        adjusted_idx = clamp(current_idx, 0, limit)
        window['-REFORM_SLICE-'].update(range=(0, limit), value=adjusted_idx)
        window['-PLANE_DESC-'].update(plane_label(plane))
        window['-REFORM_TITLE-'].update(f'再構成断面 - {plane_label(plane)}')
        return adjusted_idx

    def update_readouts(plane: str, axial_idx: int, reform_idx: int, reform_limit: int, ww: int, wl: int, *, update_status: bool = True) -> None:
        window['-AXIAL_INFO-'].update(f'{axial_idx + 1} / {z_dim}')
        window['-REFORM_INFO-'].update(f'{reform_idx + 1} / {reform_limit + 1}')
        window['-WW_INFO-'].update(str(ww))
        window['-WL_INFO-'].update(str(wl))
        if update_status:
            window['-STATUS-'].update(
                f'平面: {plane_label(plane)} | Axial {axial_idx + 1}/{z_dim} | 再構成 {reform_idx + 1}/{reform_limit + 1} | WW {ww} | WL {wl}'
            )

    def refresh_images(val_dict: Dict, *, update_status: bool = True) -> None:
        plane, axial_idx, reform_idx, ww, wl, reform_limit = parse_values(val_dict)
        reform_idx = update_reformat_slider(plane, reform_idx)
        val_dict['-REFORM_SLICE-'] = reform_idx

        axial_slice = extract_plane(volume, 'Axial', axial_idx)
        reform_slice = extract_plane(volume, plane, reform_idx)

        axial_img = window_level_image(axial_slice, ww, wl)
        reform_img = window_level_image(reform_slice, ww, wl)

        overlay = update_overlay_coords(plane, reform_idx, reform_limit, axial_img.shape)
        window['-AXIAL-'].update(data=to_pil(axial_img, size=image_size, overlay_line=overlay))
        window['-REFORM-'].update(data=to_pil(reform_img, size=image_size))
        update_readouts(plane, axial_idx, reform_idx, reform_limit, ww, wl, update_status=update_status)

    def apply_preset(key: str, val_dict: Dict) -> Tuple[Dict, str]:
        label, ww_val, wl_val = preset_map[key]
        ww_val = clamp(ww_val, 1, ww_slider_max)
        wl_val = clamp(wl_val, wl_min, wl_max)
        window['-WW-'].update(value=ww_val)
        window['-WL-'].update(value=wl_val)
        val_dict['-WW-'] = ww_val
        val_dict['-WL-'] = wl_val
        window['-PRESET_INFO-'].update(f'{label}: WW {ww_val} / WL {wl_val}')
        return val_dict, label

    def reset_plane_state(val_dict: Dict) -> Dict:
        val_dict['-PLANE-'] = default_plane
        val_dict['-REFORM_SLICE-'] = reformat_default
        window['-PLANE-'].update(value=default_plane)
        window['-REFORM_SLICE-'].update(range=(0, plane_limit(volume, default_plane)), value=reformat_default)
        window['-PLANE_DESC-'].update(plane_label(default_plane))
        window['-REFORM_TITLE-'].update(f'再構成断面 - {plane_label(default_plane)}')
        return val_dict

    init_event, init_values = window.read(timeout=0)
    if init_values is None:
        window.close()
        return
    if init_values.get('-PLANE-') is None:
        init_values['-PLANE-'] = default_plane
    refresh_images(init_values)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Quit' or values is None:
            break
        if event == '-FULLSCREEN-':
            try:
                window.maximize()
            except AttributeError:
                window.Maximize()
            window['-STATUS-'].update('全画面表示に切り替えました')
            continue
        if event == '-RESET-PLANE-':
            values = reset_plane_state(values)
            refresh_images(values)
            window['-STATUS-'].update('断面設定を初期化しました')
            continue
        if event == '-RESET-WINDOW-':
            base_ww = int(round(default_ww))
            base_wl = int(round(default_wl))
            window['-WW-'].update(value=base_ww)
            window['-WL-'].update(value=base_wl)
            values['-WW-'] = base_ww
            values['-WL-'] = base_wl
            window['-PRESET_INFO-'].update('WW/WL を初期化しました')
            refresh_images(values)
            continue
        if event in preset_map:
            values, preset_label = apply_preset(event, values)
            refresh_images(values, update_status=False)
            window['-STATUS-'].update(f'プリセット適用: {preset_label}')
            continue
        if event in ('-PLANE-', '-AXIAL_SLICE-', '-REFORM_SLICE-', '-WW-', '-WL-'):
            refresh_images(values)

    window.close()


if __name__ == '__main__':
    main()
