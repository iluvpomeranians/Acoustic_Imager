"""
Click and mouse handlers for menu, buttons, and gallery.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import time

from . import ui_cache
from .button import buttons, menu_buttons
from .gallery import get_displayed_gallery_items, get_folder_displayed_items, _viewer_rubber_band_offset, _video_read_frame_at
from .archive_panel import load_archive_folders, add_folder, rename_folder as archive_rename_folder, delete_folder as archive_delete_folder, move_files_to_folder, remove_files_from_all_folders, save_archive_folders
from .viewer_dock import trigger_viewer_button_feedback
from .menu import get_recording_timestamp_rect
from .screenshot import save_screenshot
from ..config import SOURCE_MODES, SOURCE_DEFAULT
from ..state import button_state
from ..io.gallery_metadata import save_metadata
from .video_recorder import VideoRecorder


def _persist_gallery_metadata(output_dir: Optional[Path]) -> None:
    """Write current tags, priority, and tag_data to gallery_metadata.json."""
    if not output_dir:
        return
    save_metadata(
        output_dir,
        getattr(button_state, "gallery_file_priorities", {}),
        getattr(button_state, "gallery_file_tags", {}),
        getattr(button_state, "gallery_tag_data", {}),
    )


def _close_select_mode_modals() -> None:
    """Close all panels that are only visible in select mode."""
    button_state.gallery_priority_modal_open = False
    button_state.gallery_tags_modal_open = False
    button_state.gallery_rename_modal_open = False
    button_state.gallery_tag_modal_open = False
    button_state.gallery_tag_active_field = ""
    button_state.gallery_archive_move_modal_open = False
    button_state.gallery_archive_folder_action_id = None
    button_state.gallery_archive_rename_folder_id = None
    button_state.gallery_archive_rename_query = ""
    button_state.gallery_archive_delete_confirm_folder_id = None


def _get_current_gallery_items(output_dir: Optional[Path]):
    """Return items for current view (main gallery or folder)."""
    if not output_dir:
        return []
    view_id = getattr(button_state, "gallery_archive_folder_view_id", None)
    return get_folder_displayed_items(output_dir, view_id) if view_id else get_displayed_gallery_items(output_dir)


def _apply_rename(output_dir: Optional[Path]) -> None:
    """Rename the selected file(s) using gallery_rename_query as the new stem.
    Single selection: name becomes stem + extension.
    Multiple: names become stem_1, stem_2, ... (same extension). No-op if none selected."""
    if not output_dir:
        return
    new_stem = (button_state.gallery_rename_query or "").strip()
    if not new_stem:
        return
    items = _get_current_gallery_items(output_dir)
    sel = sorted(button_state.gallery_selected_items)
    for i, idx in enumerate(sel):
        if idx >= len(items):
            continue
        old_path = items[idx][0]
        if len(sel) == 1:
            new_name = new_stem + old_path.suffix
        else:
            new_name = f"{new_stem}_{i + 1}{old_path.suffix}"
        new_path = old_path.parent / new_name
        if new_path.exists() and new_path != old_path:
            continue
        try:
            old_path.rename(new_path)
            # Evict old thumbnail from cache
            ui_cache._THUMB_CACHE.pop(old_path, None)
            ui_cache._THUMB_CACHE_MTIME.pop(old_path, None)
            # Transfer priority/tags to new filename
            old_fn = old_path.name
            new_fn = new_path.name
            prios = getattr(button_state, 'gallery_file_priorities', {})
            if old_fn in prios:
                prios[new_fn] = prios.pop(old_fn)
            tags = getattr(button_state, 'gallery_file_tags', {})
            if old_fn in tags:
                tags[new_fn] = tags.pop(old_fn)
        except Exception as e:
            print(f"Rename failed: {e}")
    button_state.gallery_storage_dirty = True
    _persist_gallery_metadata(output_dir)


def _open_tag_modal(output_dir: Optional[Path]) -> None:
    """Pre-fill tag field values from the first selected file and open the tag modal."""
    rects = getattr(button_state, 'gallery_thumbnail_rects', [])
    sel_paths = [r['filepath'] for r in rects if r['idx'] in button_state.gallery_selected_items]
    first_path = sel_paths[0] if sel_paths else None

    tag_data = getattr(button_state, 'gallery_tag_data', {})
    existing = tag_data.get(first_path.name, {}) if first_path else {}

    button_state.gallery_tag_field_values = {
        "asset_name": first_path.stem if first_path else "",
        "asset_type": existing.get("asset_type", ""),
        "leak_type":  existing.get("leak_type",  ""),
    }
    button_state.gallery_tag_active_field = ""
    button_state.gallery_tag_keyboard_query = ""
    button_state.gallery_tag_modal_open = True
    button_state.gallery_priority_modal_open = False
    button_state.gallery_rename_modal_open = False


def _commit_active_tag_field() -> None:
    """Commit current keyboard input into field_vals (in-memory only)."""
    if button_state.gallery_tag_active_field:
        button_state.gallery_tag_field_values[button_state.gallery_tag_active_field] = (
            button_state.gallery_tag_keyboard_query
        )


def _apply_tag_save(output_dir: Optional[Path]) -> None:
    """Commit tag modal values: rename first file if Asset Name changed; apply Asset Type and Leak Type to all selected.
    Multi-select: same tags (asset_type, leak_type) apply to every selected item; only the first can be renamed by name."""
    if not output_dir:
        return
    # Commit any active field first
    _commit_active_tag_field()
    button_state.gallery_tag_active_field = ""

    field_vals = getattr(button_state, 'gallery_tag_field_values', {})
    new_asset_name = field_vals.get("asset_name", "").strip()
    new_asset_type = field_vals.get("asset_type", "").strip()
    new_leak_type  = field_vals.get("leak_type",  "").strip()

    rects = getattr(button_state, 'gallery_thumbnail_rects', [])
    sel_rects = [r for r in rects if r['idx'] in button_state.gallery_selected_items]
    if not sel_rects:
        return

    tag_data = getattr(button_state, 'gallery_tag_data', {})
    first_path: Optional[Path] = sel_rects[0]['filepath']
    effective_name = first_path.name

    # Rename first selected file if Asset Name changed
    if new_asset_name and new_asset_name != first_path.stem:
        new_file = first_path.parent / (new_asset_name + first_path.suffix)
        if not (new_file.exists() and new_file != first_path):
            try:
                first_path.rename(new_file)
                # Transfer data to new key
                if first_path.name in tag_data:
                    tag_data[new_file.name] = tag_data.pop(first_path.name)
                prios = getattr(button_state, 'gallery_file_priorities', {})
                if first_path.name in prios:
                    prios[new_file.name] = prios.pop(first_path.name)
                file_tags = getattr(button_state, "gallery_file_tags", {})
                if first_path.name in file_tags:
                    file_tags[new_file.name] = file_tags.pop(first_path.name)
                ui_cache._THUMB_CACHE.pop(first_path, None)
                ui_cache._THUMB_CACHE_MTIME.pop(first_path, None)
                effective_name = new_file.name
            except Exception as e:
                print(f"Tag save rename failed: {e}")

    # Save asset_type + leak_type for ALL selected files
    for rect in sel_rects:
        fp: Path = rect['filepath']
        fname = effective_name if fp == first_path else fp.name
        entry = tag_data.get(fname, {})
        entry["asset_type"] = new_asset_type
        entry["leak_type"]  = new_leak_type
        tag_data[fname] = entry

    button_state.gallery_tag_data = tag_data
    button_state.gallery_storage_dirty = True
    _persist_gallery_metadata(output_dir)


def handle_gallery_click(x: int, y: int, output_dir: Optional[Path]) -> bool:
    """
    Handle clicks in gallery view.
    Returns True if click was handled in gallery, False otherwise.
    """
    if not button_state.gallery_open:
        return False

    if "gallery_back" in menu_buttons and menu_buttons["gallery_back"].contains(x, y):
        trigger_viewer_button_feedback("gallery_back")
        if button_state.gallery_viewer_mode == "grid":
            if getattr(button_state, "gallery_archive_folder_view_id", None):
                # In folder view: go back to main gallery
                button_state.gallery_archive_folder_view_id = None
                button_state.gallery_scroll_offset = 0
                button_state.gallery_selected_item = None
                button_state.gallery_selected_items.clear()
            else:
                # In main gallery: close gallery
                button_state.gallery_open = False
                button_state.gallery_scroll_offset = 0
                button_state.gallery_selected_item = None
                button_state.gallery_archive_folder_action_id = None
                button_state.gallery_archive_rename_folder_id = None
                button_state.gallery_archive_rename_query = ""
                button_state.gallery_archive_delete_confirm_folder_id = None
                button_state.gallery_select_mode = False
                button_state.gallery_selected_items.clear()
                _close_select_mode_modals()

        else:
            button_state.gallery_viewer_mode = "grid"
            button_state.gallery_video_playing = False
            button_state.gallery_video_frame_idx = 0
            button_state.gallery_tag_info_open = False

        return True

    if button_state.gallery_archive_move_modal_open:
        folders = getattr(button_state, "gallery_archive_folders", [])
        view_id = getattr(button_state, "gallery_archive_folder_view_id", None)
        items = get_folder_displayed_items(output_dir, view_id) if view_id else get_displayed_gallery_items(output_dir)
        filenames = [items[i][0].name for i in button_state.gallery_selected_items if i < len(items)]

        if "move_to_gallery" in menu_buttons and menu_buttons["move_to_gallery"].contains(x, y):
            if output_dir and filenames and view_id:
                button_state.gallery_archive_folders = remove_files_from_all_folders(
                    output_dir, folders, filenames
                )
            button_state.gallery_selected_items.clear()
            button_state.gallery_archive_move_modal_open = False
            return True

        for folder in folders:
            key = "move_to_folder_" + folder.get("id", "")
            if key in menu_buttons and menu_buttons[key].contains(x, y):
                if output_dir and filenames:
                    button_state.gallery_archive_folders = move_files_to_folder(
                        output_dir, folders, folder["id"], filenames
                    )
                button_state.gallery_selected_items.clear()
                button_state.gallery_archive_move_modal_open = False
                return True
        # Click outside folder buttons closes modal
        button_state.gallery_archive_move_modal_open = False
        return True

    if button_state.gallery_delete_modal_open:
        if "modal_yes" in menu_buttons and menu_buttons["modal_yes"].contains(x, y):
            view_id = getattr(button_state, "gallery_archive_folder_view_id", None)
            items = get_folder_displayed_items(output_dir, view_id) if view_id else get_displayed_gallery_items(output_dir)

            try:
                if button_state.gallery_delete_modal_kind == "batch":
                    deleted = 0
                    prios = getattr(button_state, "gallery_file_priorities", {})
                    tags = getattr(button_state, "gallery_file_tags", {})
                    tag_data = getattr(button_state, "gallery_tag_data", {})
                    for idx in sorted(button_state.gallery_selected_items, reverse=True):
                        if 0 <= idx < len(items):
                            path = items[idx][0]
                            try:
                                path.unlink()
                                deleted += 1
                                fn = path.name
                                prios.pop(fn, None)
                                tags.pop(fn, None)
                                tag_data.pop(fn, None)
                            except Exception as e:
                                print(f"Failed to delete {path}: {e}")

                    button_state.gallery_selected_items.clear()
                    button_state.gallery_select_mode = False
                    button_state.gallery_delete_modal_open = False
                    button_state.gallery_delete_modal_kind = "single"
                    if deleted > 0:
                        button_state.gallery_storage_dirty = True
                        _persist_gallery_metadata(output_dir)
                    return True

                if button_state.gallery_selected_item is not None and button_state.gallery_selected_item < len(items):
                    filepath = items[button_state.gallery_selected_item][0]
                    filepath.unlink()
                    fn = filepath.name
                    prios = getattr(button_state, "gallery_file_priorities", {})
                    tags = getattr(button_state, "gallery_file_tags", {})
                    tag_data = getattr(button_state, "gallery_tag_data", {})
                    prios.pop(fn, None)
                    tags.pop(fn, None)
                    tag_data.pop(fn, None)
                    button_state.gallery_storage_dirty = True
                    _persist_gallery_metadata(output_dir)

                    if len(items) <= 1:
                        button_state.gallery_viewer_mode = "grid"
                        button_state.gallery_selected_item = None
                    else:
                        if button_state.gallery_selected_item >= len(items) - 1:
                            button_state.gallery_selected_item = len(items) - 2

                button_state.gallery_delete_modal_open = False
                button_state.gallery_delete_modal_kind = "single"
            except Exception:
                button_state.gallery_delete_modal_open = False
                button_state.gallery_delete_modal_kind = "single"
            return True

        if "modal_no" in menu_buttons and menu_buttons["modal_no"].contains(x, y):
            button_state.gallery_delete_modal_open = False
            button_state.gallery_delete_modal_kind = "single"
            return True

        return True

    # Move to button (select mode only)
    if button_state.gallery_viewer_mode == "grid" and button_state.gallery_select_mode:
        if "gallery_move_to" in menu_buttons and menu_buttons["gallery_move_to"].contains(x, y):
            if button_state.gallery_selected_items:
                folders = getattr(button_state, "gallery_archive_folders", [])
                if folders:
                    button_state.gallery_archive_move_modal_open = True
                else:
                    button_state.gallery_archive_move_hint_until = time.time() + 2.5
            return True

    # Archive folder delete confirmation modal
    if getattr(button_state, "gallery_archive_delete_confirm_folder_id", None):
        folder_id = button_state.gallery_archive_delete_confirm_folder_id
        if "archive_delete_yes" in menu_buttons and menu_buttons["archive_delete_yes"].contains(x, y):
            if output_dir and folder_id:
                folders = getattr(button_state, "gallery_archive_folders", [])
                button_state.gallery_archive_folders = archive_delete_folder(output_dir, folders, folder_id)
            button_state.gallery_archive_delete_confirm_folder_id = None
            button_state.gallery_archive_folder_action_id = None
            if getattr(button_state, "gallery_archive_folder_view_id", None) == folder_id:
                button_state.gallery_archive_folder_view_id = None
            return True
        if "archive_delete_no" in menu_buttons and menu_buttons["archive_delete_no"].contains(x, y):
            button_state.gallery_archive_delete_confirm_folder_id = None
            return True
        return True

    # Archive folder action modal (Rename/Delete when in select mode) and rename sub-modal
    folder_action_id = getattr(button_state, "gallery_archive_folder_action_id", None)
    if folder_action_id:
        in_rename = getattr(button_state, "gallery_archive_rename_folder_id", None) == folder_action_id
        if in_rename:
            # Rename mode: keyboard (tag-style), Save, Cancel
            for c in "abcdefghijklmnopqrstuvwxyz0123456789":
                key = f"archive_rename_key_{c}"
                if key in menu_buttons and menu_buttons[key].contains(x, y):
                    button_state.gallery_archive_rename_query = (getattr(button_state, "gallery_archive_rename_query", "") or "") + c
                    return True
            if "archive_rename_key_backspace" in menu_buttons and menu_buttons["archive_rename_key_backspace"].contains(x, y):
                q = getattr(button_state, "gallery_archive_rename_query", "") or ""
                button_state.gallery_archive_rename_query = q[:-1]
                return True
            if "archive_rename_key_clear" in menu_buttons and menu_buttons["archive_rename_key_clear"].contains(x, y):
                button_state.gallery_archive_rename_query = ""
                return True
            if "archive_rename_key_done" in menu_buttons and menu_buttons["archive_rename_key_done"].contains(x, y):
                new_name = (getattr(button_state, "gallery_archive_rename_query", "") or "").strip()
                if new_name and output_dir:
                    folders = getattr(button_state, "gallery_archive_folders", [])
                    button_state.gallery_archive_folders = archive_rename_folder(
                        output_dir, folders, folder_action_id, new_name
                    )
                button_state.gallery_archive_rename_folder_id = None
                button_state.gallery_archive_rename_query = ""
                return True
            if "archive_rename_save" in menu_buttons and menu_buttons["archive_rename_save"].contains(x, y):
                new_name = (getattr(button_state, "gallery_archive_rename_query", "") or "").strip()
                if new_name and output_dir:
                    folders = getattr(button_state, "gallery_archive_folders", [])
                    button_state.gallery_archive_folders = archive_rename_folder(
                        output_dir, folders, folder_action_id, new_name
                    )
                button_state.gallery_archive_rename_folder_id = None
                button_state.gallery_archive_rename_query = ""
                return True
            if "archive_rename_cancel" in menu_buttons and menu_buttons["archive_rename_cancel"].contains(x, y):
                button_state.gallery_archive_rename_folder_id = None
                button_state.gallery_archive_rename_query = ""
                return True
        else:
            # Action mode: Rename | Delete | Cancel
            if "archive_folder_rename" in menu_buttons and menu_buttons["archive_folder_rename"].contains(x, y):
                folder = next((f for f in getattr(button_state, "gallery_archive_folders", []) if f.get("id") == folder_action_id), None)
                if folder:
                    button_state.gallery_archive_rename_folder_id = folder_action_id
                    button_state.gallery_archive_rename_query = folder.get("name", "Folder") or "Folder"
                return True
            if "archive_folder_delete" in menu_buttons and menu_buttons["archive_folder_delete"].contains(x, y):
                button_state.gallery_archive_delete_confirm_folder_id = folder_action_id
                return True
            if "archive_folder_cancel" in menu_buttons and menu_buttons["archive_folder_cancel"].contains(x, y):
                button_state.gallery_archive_folder_action_id = None
                return True
        # Absorb clicks on modal panel
        if "archive_folder_modal_panel" in menu_buttons and menu_buttons["archive_folder_modal_panel"].contains(x, y):
            return True
        # Click outside: close
        button_state.gallery_archive_folder_action_id = None
        button_state.gallery_archive_rename_folder_id = None
        button_state.gallery_archive_rename_query = ""
        button_state.gallery_archive_delete_confirm_folder_id = None
        return True

    # Archive panel (only visible in main gallery, not in folder view)
    if not getattr(button_state, "gallery_archive_folder_view_id", None) and "archive_panel" in menu_buttons and menu_buttons["archive_panel"].contains(x, y):
        if "archive_add_btn" in menu_buttons and menu_buttons["archive_add_btn"].contains(x, y):
            if output_dir:
                folders = getattr(button_state, "gallery_archive_folders", [])
                button_state.gallery_archive_folders = add_folder(output_dir, folders)
        else:
            # Check folder clicks
            for folder in getattr(button_state, "gallery_archive_folders", []):
                key = f"archive_folder_{folder.get('id', '')}"
                if key in menu_buttons and menu_buttons[key].contains(x, y):
                    if button_state.gallery_select_mode:
                        button_state.gallery_archive_folder_action_id = folder["id"]
                    else:
                        button_state.gallery_archive_folder_view_id = folder["id"]
                    return True
        return True

    # Dock row buttons have priority.
    # In select mode: Tags / Priority / Rename  (Search / Filter / Sort hidden)
    # Require at least one selected item to open; otherwise show hint and do not open.
    if button_state.gallery_viewer_mode == "grid" and button_state.gallery_select_mode:
        if "gallery_dock_tags" in menu_buttons and menu_buttons["gallery_dock_tags"].contains(x, y):
            if button_state.gallery_tag_modal_open:
                button_state.gallery_tag_modal_open = False
            elif not button_state.gallery_selected_items:
                button_state.gallery_select_first_hint_until = time.time() + 2.5
            else:
                _open_tag_modal(output_dir)
            return True
        if "gallery_dock_priority" in menu_buttons and menu_buttons["gallery_dock_priority"].contains(x, y):
            if button_state.gallery_priority_modal_open:
                button_state.gallery_priority_modal_open = False
            elif not button_state.gallery_selected_items:
                button_state.gallery_select_first_hint_until = time.time() + 2.5
            else:
                button_state.gallery_priority_modal_open = True
                button_state.gallery_tag_modal_open = False
                button_state.gallery_tags_modal_open = False
                button_state.gallery_rename_modal_open = False
            return True
        if "gallery_dock_rename" in menu_buttons and menu_buttons["gallery_dock_rename"].contains(x, y):
            if button_state.gallery_rename_modal_open:
                button_state.gallery_rename_modal_open = False
            elif not button_state.gallery_selected_items:
                button_state.gallery_select_first_hint_until = time.time() + 2.5
            else:
                items = _get_current_gallery_items(output_dir)
                sel = sorted(button_state.gallery_selected_items)
                if sel and sel[0] < len(items):
                    first_path = items[sel[0]][0]
                    button_state.gallery_rename_query = first_path.stem
                else:
                    button_state.gallery_rename_query = ""
                button_state.gallery_rename_modal_open = True
                button_state.gallery_tag_modal_open = False
                button_state.gallery_tags_modal_open = False
                button_state.gallery_priority_modal_open = False
            return True

    # Priority modal option clicks
    if button_state.gallery_priority_modal_open:
        for value in ("high", "medium", "low"):
            key = f"gallery_priority_opt_{value}"
            if key in menu_buttons and menu_buttons[key].contains(x, y):
                items = _get_current_gallery_items(output_dir)
                priorities = getattr(button_state, 'gallery_file_priorities', {})
                for idx in button_state.gallery_selected_items:
                    if idx < len(items):
                        fname = items[idx][0].name
                        # Toggle: clicking the same priority again removes it
                        if priorities.get(fname) == value:
                            priorities[fname] = ""
                        else:
                            priorities[fname] = value
                button_state.gallery_file_priorities = priorities
                _persist_gallery_metadata(output_dir)
                return True
        if "gallery_priority_modal_panel" in menu_buttons and menu_buttons["gallery_priority_modal_panel"].contains(x, y):
            return True
        button_state.gallery_priority_modal_open = False

    # Rename keyboard clicks
    if button_state.gallery_rename_modal_open:
        if "rename_key_done" in menu_buttons and menu_buttons["rename_key_done"].contains(x, y):
            _apply_rename(output_dir)
            button_state.gallery_rename_modal_open = False
            return True
        if "rename_key_clear" in menu_buttons and menu_buttons["rename_key_clear"].contains(x, y):
            button_state.gallery_rename_query = ""
            return True
        if "rename_key_backspace" in menu_buttons and menu_buttons["rename_key_backspace"].contains(x, y):
            button_state.gallery_rename_query = (button_state.gallery_rename_query or "")[:-1]
            return True
        for c in "abcdefghijklmnopqrstuvwxyz0123456789":
            key = f"rename_key_{c}"
            if key in menu_buttons and menu_buttons[key].contains(x, y):
                button_state.gallery_rename_query = (button_state.gallery_rename_query or "") + c
                return True
        if "rename_keyboard_panel" in menu_buttons and menu_buttons["rename_keyboard_panel"].contains(x, y):
            return True
        button_state.gallery_rename_modal_open = False
        return True

    # ── Tag edit modal (keyboard merged; auto-save on field switch / Done / close) ─
    if button_state.gallery_tag_modal_open:
        # Tag keyboard keys (always visible when modal open)
        if button_state.gallery_tag_active_field:
            for c in "abcdefghijklmnopqrstuvwxyz0123456789":
                bk = f"tag_key_{c}"
                if bk in menu_buttons and menu_buttons[bk].contains(x, y):
                    button_state.gallery_tag_keyboard_query = (
                        button_state.gallery_tag_keyboard_query or "") + c
                    return True
            if "tag_key_backspace" in menu_buttons and menu_buttons["tag_key_backspace"].contains(x, y):
                button_state.gallery_tag_keyboard_query = (
                    button_state.gallery_tag_keyboard_query or "")[:-1]
                return True
            if "tag_key_clear" in menu_buttons and menu_buttons["tag_key_clear"].contains(x, y):
                button_state.gallery_tag_keyboard_query = ""
                return True
            if "tag_key_done" in menu_buttons and menu_buttons["tag_key_done"].contains(x, y):
                _apply_tag_save(output_dir)
                button_state.gallery_tag_modal_open = False
                button_state.gallery_tag_active_field = ""
                return True
        if "tag_keyboard_panel" in menu_buttons and menu_buttons["tag_keyboard_panel"].contains(x, y):
            return True

        # Field clicks → switch field (auto-save current first)
        for fkey, flabel, _ in [("asset_name", "Asset Name", ""), ("asset_type", "Asset Type", ""), ("leak_type", "Leak Type", "")]:
            btn_key = f"tag_field_{fkey}"
            if btn_key in menu_buttons and menu_buttons[btn_key].contains(x, y):
                _apply_tag_save(output_dir)  # auto-save before switching
                if fkey == "asset_name":
                    rects = getattr(button_state, 'gallery_thumbnail_rects', [])
                    sel_paths = [r['filepath'] for r in rects if r['idx'] in button_state.gallery_selected_items]
                    default_stem = sel_paths[0].stem if sel_paths else ""
                    prefill = button_state.gallery_tag_field_values.get("asset_name", default_stem)
                else:
                    prefill = button_state.gallery_tag_field_values.get(fkey, "")
                button_state.gallery_tag_active_field = fkey
                button_state.gallery_tag_keyboard_query = prefill
                return True

        # Absorb clicks on modal panel
        if "tag_modal_panel" in menu_buttons and menu_buttons["tag_modal_panel"].contains(x, y):
            return True
        # Click outside modal: auto-save and close
        _apply_tag_save(output_dir)
        button_state.gallery_tag_modal_open = False
        button_state.gallery_tag_active_field = ""
        return True

    # ── Viewer tag info panel ─────────────────────────────────────────────────
    if button_state.gallery_viewer_mode in ("image", "video"):
        if "tag_info_close" in menu_buttons and menu_buttons["tag_info_close"].contains(x, y):
            button_state.gallery_tag_info_open = False
            return True
        if "tag_info_panel" in menu_buttons and menu_buttons["tag_info_panel"].contains(x, y):
            return True
        if "viewer_tag_btn" in menu_buttons and menu_buttons["viewer_tag_btn"].contains(x, y):
            button_state.gallery_tag_info_open = not button_state.gallery_tag_info_open
            return True

    # Search / Filter / Sort (only in normal grid mode, not select mode)
    if button_state.gallery_viewer_mode == "grid" and not button_state.gallery_select_mode:
        if "gallery_dock_search" in menu_buttons and menu_buttons["gallery_dock_search"].contains(x, y):
            button_state.gallery_search_keyboard_open = not button_state.gallery_search_keyboard_open
            if button_state.gallery_search_keyboard_open:
                button_state.gallery_filter_modal_open = False
                button_state.gallery_sort_modal_open = False
            return True
        if "gallery_dock_filter" in menu_buttons and menu_buttons["gallery_dock_filter"].contains(x, y):
            button_state.gallery_filter_modal_open = not button_state.gallery_filter_modal_open
            if button_state.gallery_filter_modal_open:
                button_state.gallery_sort_modal_open = False
                button_state.gallery_search_keyboard_open = False
            return True
        if "gallery_dock_sort" in menu_buttons and menu_buttons["gallery_dock_sort"].contains(x, y):
            button_state.gallery_sort_modal_open = not button_state.gallery_sort_modal_open
            if button_state.gallery_sort_modal_open:
                button_state.gallery_filter_modal_open = False
                button_state.gallery_search_keyboard_open = False
            return True

    # Filter modal: option click only sets filter (modal stays open); click outside closes and fall through
    if button_state.gallery_filter_modal_open:
        for value in ("all", "image", "video"):
            key = f"gallery_filter_opt_{value}"
            if key in menu_buttons and menu_buttons[key].contains(x, y):
                button_state.gallery_filter_type = value
                return True
        if "gallery_filter_modal_panel" in menu_buttons and menu_buttons["gallery_filter_modal_panel"].contains(x, y):
            return True
        button_state.gallery_filter_modal_open = False
        # fall through so the same click can hit thumbnails or other controls

    # Sort modal: option click only sets sort (modal stays open); click outside closes and fall through
    if button_state.gallery_sort_modal_open:
        for value in ("date", "name", "size", "priority"):
            key = f"gallery_sort_opt_{value}"
            if key in menu_buttons and menu_buttons[key].contains(x, y):
                button_state.gallery_sort_by = value
                return True
        if "gallery_sort_modal_panel" in menu_buttons and menu_buttons["gallery_sort_modal_panel"].contains(x, y):
            return True
        button_state.gallery_sort_modal_open = False
        # fall through so the same click can hit thumbnails or other controls

    if button_state.gallery_search_keyboard_open:
        if "search_key_done" in menu_buttons and menu_buttons["search_key_done"].contains(x, y):
            button_state.gallery_search_keyboard_open = False
            return True
        if "search_key_clear" in menu_buttons and menu_buttons["search_key_clear"].contains(x, y):
            button_state.gallery_search_query = ""
            return True
        if "search_key_backspace" in menu_buttons and menu_buttons["search_key_backspace"].contains(x, y):
            button_state.gallery_search_query = (button_state.gallery_search_query or "")[:-1]
            return True
        for c in "abcdefghijklmnopqrstuvwxyz0123456789":
            key = f"search_key_{c}"
            if key in menu_buttons and menu_buttons[key].contains(x, y):
                button_state.gallery_search_query = (button_state.gallery_search_query or "") + c
                return True
        if "search_keyboard_panel" in menu_buttons and menu_buttons["search_keyboard_panel"].contains(x, y):
            return True
        button_state.gallery_search_keyboard_open = False
        return True

    if button_state.gallery_viewer_mode in ("image", "video", "grid"):
        if "gallery_delete" in menu_buttons and menu_buttons["gallery_delete"].contains(x, y):
            button_state.gallery_delete_modal_open = True
            return True

    if button_state.gallery_viewer_mode in ("image", "video"):
        if "gallery_prev" in menu_buttons and menu_buttons["gallery_prev"].contains(x, y):
            trigger_viewer_button_feedback("gallery_prev")
            if button_state.gallery_selected_item > 0:
                button_state.gallery_selected_item -= 1
                items = get_displayed_gallery_items(output_dir)

                if button_state.gallery_selected_item < len(items):
                    new_item_type = items[button_state.gallery_selected_item][1]
                    button_state.gallery_viewer_mode = new_item_type

                    if new_item_type == "video":
                        button_state.gallery_video_playing = False
                        button_state.gallery_video_frame_idx = 0
                # Instant transition: no swipe animation
                button_state.gallery_viewer_swipe_offset = 0.0
                button_state.gallery_viewer_swipe_velocity = 0.0
                button_state.gallery_viewer_swipe_inertia_active = False
            return True

        if "gallery_next" in menu_buttons and menu_buttons["gallery_next"].contains(x, y):
            trigger_viewer_button_feedback("gallery_next")
            items = get_displayed_gallery_items(output_dir)
            if button_state.gallery_selected_item < len(items) - 1:
                button_state.gallery_selected_item += 1

                if button_state.gallery_selected_item < len(items):
                    new_item_type = items[button_state.gallery_selected_item][1]
                    button_state.gallery_viewer_mode = new_item_type

                    if new_item_type == "video":
                        button_state.gallery_video_playing = False
                        button_state.gallery_video_frame_idx = 0
                # Instant transition: no swipe animation
                button_state.gallery_viewer_swipe_offset = 0.0
                button_state.gallery_viewer_swipe_velocity = 0.0
                button_state.gallery_viewer_swipe_inertia_active = False
            return True

    if button_state.gallery_viewer_mode == "video":
        if "gallery_play" in menu_buttons and menu_buttons["gallery_play"].contains(x, y):
            trigger_viewer_button_feedback("gallery_play")
            button_state.gallery_video_playing = not button_state.gallery_video_playing
            return True

        if "gallery_progress" in menu_buttons and menu_buttons["gallery_progress"].contains(x, y):
            items = get_displayed_gallery_items(output_dir)
            if button_state.gallery_selected_item is not None and button_state.gallery_selected_item < len(items):
                filepath = items[button_state.gallery_selected_item][0]
                _, total_frames, _ = _video_read_frame_at(filepath, 0)
                if total_frames > 0:
                    progress_btn = menu_buttons["gallery_progress"]
                    click_pos = (x - progress_btn.x) / progress_btn.w
                    click_pos = max(0.0, min(1.0, click_pos))

                    button_state.gallery_video_frame_idx = int(total_frames * click_pos)
                    button_state.gallery_video_playing = False
            return True

    if button_state.gallery_viewer_mode == "grid":

        if "gallery_select_mode" in menu_buttons:
            btn = menu_buttons["gallery_select_mode"]
            if btn.contains(x, y):
                button_state.gallery_select_mode = not button_state.gallery_select_mode
                if not button_state.gallery_select_mode:
                    button_state.gallery_selected_items.clear()
                    _close_select_mode_modals()
                return True

        if button_state.gallery_select_mode and "gallery_select_all" in menu_buttons:
            if menu_buttons["gallery_select_all"].contains(x, y):
                items = get_displayed_gallery_items(output_dir)
                if items:
                    all_selected = len(button_state.gallery_selected_items) == len(items)
                    if all_selected:
                        button_state.gallery_selected_items.clear()
                    else:
                        button_state.gallery_selected_items = set(range(len(items)))

                return True

        if button_state.gallery_select_mode and "gallery_delete_selected" in menu_buttons:
            if menu_buttons["gallery_delete_selected"].contains(x, y):
                if button_state.gallery_selected_items:
                    button_state.gallery_delete_modal_open = True
                    button_state.gallery_delete_modal_kind = "batch"
                return True

        if hasattr(button_state, 'gallery_thumbnail_rects'):
            for thumb in button_state.gallery_thumbnail_rects:
                if (thumb['x'] <= x <= thumb['x'] + thumb['w'] and
                    thumb['y'] <= y <= thumb['y'] + thumb['h']):
                    idx = thumb['idx']

                    if button_state.gallery_select_mode:
                        if idx in button_state.gallery_selected_items:
                            button_state.gallery_selected_items.remove(idx)
                        else:
                            button_state.gallery_selected_items.add(idx)
                    else:
                        button_state.gallery_selected_item = idx
                        item_type = thumb['type']
                        if item_type == "image":
                            button_state.gallery_viewer_mode = "image"
                        else:
                            button_state.gallery_viewer_mode = "video"
                            button_state.gallery_video_playing = False
                            button_state.gallery_video_frame_idx = 0
                    return True

    return True


def handle_menu_click(
    x: int,
    y: int,
    current_frame: Optional[np.ndarray],
    output_dir: Optional[Path],
    video_recorder: Optional[VideoRecorder],
    width: int,
    height: int,
) -> Optional[VideoRecorder]:
    """
    Returns (possibly newly created) video_recorder, same logic as original.
    """
    if "menu" not in menu_buttons:
        return video_recorder

    if button_state.is_recording and video_recorder is not None:
        rect = get_recording_timestamp_rect()
        if rect is not None:
            rx, ry, rw, rh = rect
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                button_state.is_paused = not button_state.is_paused
                if button_state.is_paused:
                    video_recorder.pause_recording()
                else:
                    video_recorder.resume_recording()
                return video_recorder

    if menu_buttons["menu"].contains(x, y):
        button_state.menu_open = not button_state.menu_open
        return video_recorder

    # Bottom HUD pills (shot, rec, gallery) are independent of menu – handle regardless of menu_open
    if "shot" in menu_buttons and menu_buttons["shot"].w > 0 and menu_buttons["shot"].contains(x, y):
        if current_frame is not None and output_dir is not None:
            save_screenshot(current_frame, output_dir)
            button_state.gallery_storage_dirty = True
        return video_recorder

    if "gallery" in menu_buttons and menu_buttons["gallery"].w > 0 and menu_buttons["gallery"].contains(x, y):
        if button_state.is_recording and video_recorder is not None:
            video_recorder.stop_recording()
            button_state.is_recording = False
            button_state.is_paused = False
            button_state.gallery_storage_dirty = True
        button_state.gallery_open = True
        button_state.menu_open = False
        button_state.gallery_storage_dirty = True
        if output_dir:
            button_state.gallery_archive_folders = load_archive_folders(output_dir)
        return video_recorder

    if button_state.is_recording and button_state.is_paused:
        if "rec_resume" in menu_buttons and menu_buttons["rec_resume"].w > 0 and menu_buttons["rec_resume"].contains(x, y):
            if video_recorder is not None:
                video_recorder.resume_recording()
                button_state.is_paused = False
            return video_recorder
        if "rec_stop" in menu_buttons and menu_buttons["rec_stop"].w > 0 and menu_buttons["rec_stop"].contains(x, y):
            if video_recorder is not None:
                video_recorder.stop_recording()
                button_state.is_recording = False
                button_state.is_paused = False
                button_state.gallery_storage_dirty = True
            return video_recorder

    if "rec" in menu_buttons and menu_buttons["rec"].w > 0 and menu_buttons["rec"].contains(x, y):
        if video_recorder is None and output_dir is not None:
            video_recorder = VideoRecorder(output_dir, width, height, fps=30)
        if video_recorder is None:
            return None
        if not button_state.is_recording:
            if video_recorder.start_recording():
                button_state.is_recording = True
                button_state.is_paused = False
        else:
            button_state.is_paused = True
            video_recorder.pause_recording()
        return video_recorder

    # Dropdown items (fps, gain, etc.) only when menu is open
    if not button_state.menu_open:
        return video_recorder

    if "fps30" in menu_buttons and menu_buttons["fps30"].contains(x, y):
        button_state.fps_mode = "30"
        return video_recorder
    if "fps60" in menu_buttons and menu_buttons["fps60"].contains(x, y):
        button_state.fps_mode = "60"
        return video_recorder
    if "fpsmax" in menu_buttons and menu_buttons["fpsmax"].contains(x, y):
        button_state.fps_mode = "MAX"
        return video_recorder

    if "gain" in menu_buttons and menu_buttons["gain"].contains(x, y):
        button_state.gain_mode = "HIGH" if button_state.gain_mode == "LOW" else "LOW"
        menu_buttons["gain"].text = f"GAIN: {button_state.gain_mode}"
        return video_recorder

    if "colormap" in menu_buttons and menu_buttons["colormap"].contains(x, y):
        colormaps = ["MAGMA", "JET", "TURBO", "INFERNO"]
        cur = button_state.colormap_mode
        try:
            i = colormaps.index(cur)
        except ValueError:
            i = 0
        button_state.colormap_mode = colormaps[(i + 1) % len(colormaps)]
        menu_buttons["colormap"].text = f"COLOUR: {button_state.colormap_mode}"
        return video_recorder

    if "cam" in menu_buttons and menu_buttons["cam"].contains(x, y):
        button_state.camera_enabled = not button_state.camera_enabled
        menu_buttons["cam"].text = "CAM: ON" if button_state.camera_enabled else "CAM: OFF"
        return video_recorder

    if "source" in menu_buttons and menu_buttons["source"].contains(x, y):
        modes = list(SOURCE_MODES)
        cur = button_state.source_mode
        try:
            i = modes.index(cur)
        except ValueError:
            i = modes.index(SOURCE_DEFAULT)
        button_state.source_mode = modes[(i + 1) % len(modes)]
        menu_buttons["source"].text = f"SRC: {button_state.source_mode}"
        return video_recorder

    if "debug" in menu_buttons and menu_buttons["debug"].contains(x, y):
        button_state.debug_enabled = not button_state.debug_enabled
        return video_recorder

    return video_recorder


def handle_button_click(
    x: int,
    y: int,
    current_frame: Optional[np.ndarray],
    output_dir: Optional[Path],
    camera_available: bool,
    video_recorder: Optional[VideoRecorder],
    width: int,
    height: int,
) -> Optional[VideoRecorder]:
    """
    Handles button clicks:
      - Calls handle_menu_click first
      - Camera toggle
      - Source toggle (SIM <-> SPI) and calls optional callbacks
      - Debug toggle

    Returns updated video_recorder (may be created by menu click).
    """
    video_recorder = handle_menu_click(
        x, y,
        current_frame=current_frame,
        output_dir=output_dir,
        video_recorder=video_recorder,
        width=width,
        height=height,
    )

    if "source" in buttons and buttons["source"].contains(x, y):
        modes = list(SOURCE_MODES)
        cur = button_state.source_mode
        try:
            i = modes.index(cur)
        except ValueError:
            i = modes.index(SOURCE_DEFAULT)

        button_state.source_mode = modes[(i + 1) % len(modes)]
        buttons["source"].text = f"Source: {button_state.source_mode}"
        return video_recorder

    if "debug" in buttons and buttons["debug"].contains(x, y):
        button_state.debug_enabled = not button_state.debug_enabled
        buttons["debug"].is_active = button_state.debug_enabled
        buttons["debug"].text = "DEBUG: ON" if button_state.debug_enabled else "DEBUG: OFF"
        return video_recorder

    return video_recorder


def handle_gallery_viewer_mouse(event, x, y, flags, output_dir) -> bool:
    """Handle horizontal swipe/drag with inertia in image or video viewer."""
    if not button_state.gallery_open or button_state.gallery_viewer_mode not in ("image", "video"):
        return False

    now = time.perf_counter()
    items = get_displayed_gallery_items(output_dir)
    n = len(items)
    idx = button_state.gallery_selected_item
    if idx is None or n == 0:
        return False

    # Progress bar scrub (video only): takes priority over swipe on bar
    if event == cv2.EVENT_LBUTTONDOWN and button_state.gallery_viewer_mode == "video":
        if "gallery_progress" in menu_buttons and menu_buttons["gallery_progress"].contains(x, y):
            button_state.gallery_progress_dragging = True
            total_frames = getattr(button_state, "_gallery_video_total_frames", 0)
            if total_frames > 0:
                prog = menu_buttons["gallery_progress"]
                t = (x - prog.x) / prog.w
                t = max(0.0, min(1.0, t))
                button_state.gallery_video_frame_idx = int(t * (total_frames - 1))
            return True

    if event == cv2.EVENT_MOUSEMOVE and button_state.gallery_progress_dragging:
        total_frames = getattr(button_state, "_gallery_video_total_frames", 0)
        if total_frames > 0 and "gallery_progress" in menu_buttons:
            prog = menu_buttons["gallery_progress"]
            t = (x - prog.x) / prog.w
            t = max(0.0, min(1.0, t))
            button_state.gallery_video_frame_idx = int(t * (total_frames - 1))
        return True

    if event == cv2.EVENT_LBUTTONUP and button_state.gallery_progress_dragging:
        button_state.gallery_progress_dragging = False
        return True

    # LBUTTONUP: clear drag state and treat as click if we didn't move (nav/back/delete get priority)
    if event == cv2.EVENT_LBUTTONUP:
        button_state.gallery_viewer_swipe_dragging = False
        if not button_state.gallery_viewer_swipe_drag_moved:
            return handle_gallery_click(x, y, output_dir)
        off = button_state.gallery_viewer_swipe_offset
        vel = button_state.gallery_viewer_swipe_velocity
        th = getattr(button_state, "_gallery_swipe_threshold_px", ui_cache.VIEWER_SWIPE_THRESHOLD_PX)
        vth = ui_cache.VIEWER_SWIPE_VELOCITY_THRESHOLD
        # Swipe right (positive off/vel) -> prev (newer). Swipe left (negative) -> next (older).
        if (off >= th or vel >= vth) and idx > 0:
            button_state.gallery_selected_item = idx - 1
            button_state.gallery_viewer_mode = items[idx - 1][1]
            button_state.gallery_viewer_swipe_offset = 0.0
            button_state.gallery_viewer_swipe_velocity = 0.0
            if button_state.gallery_viewer_mode == "video":
                button_state.gallery_video_playing = False
                button_state.gallery_video_frame_idx = 0
            return True
        if (off <= -th or vel <= -vth) and idx < n - 1:
            button_state.gallery_selected_item = idx + 1
            button_state.gallery_viewer_mode = items[idx + 1][1]
            button_state.gallery_viewer_swipe_offset = 0.0
            button_state.gallery_viewer_swipe_velocity = 0.0
            if button_state.gallery_viewer_mode == "video":
                button_state.gallery_video_playing = False
                button_state.gallery_video_frame_idx = 0
            return True
        if abs(vel) > ui_cache.VIEWER_SWIPE_STOP_VELOCITY:
            button_state.gallery_viewer_swipe_inertia_active = True
            button_state.gallery_viewer_swipe_last_inertia_t = now
        else:
            button_state.gallery_viewer_swipe_offset = 0.0
            button_state.gallery_viewer_swipe_velocity = 0.0
        return True

    # LBUTTONDOWN: don't start swipe if on a button so nav/back/delete take priority
    if event == cv2.EVENT_LBUTTONDOWN:
        on_button = False
        for key in ("gallery_back", "gallery_prev", "gallery_next", "gallery_delete", "gallery_play", "gallery_progress"):
            if key in menu_buttons and menu_buttons[key].contains(x, y):
                on_button = True
                break
        if on_button:
            button_state.gallery_viewer_swipe_drag_moved = False
            return True
        button_state.gallery_viewer_swipe_dragging = True
        button_state.gallery_viewer_swipe_start_x = x
        button_state.gallery_viewer_swipe_offset = 0.0
        button_state.gallery_viewer_swipe_velocity = 0.0
        button_state.gallery_viewer_swipe_last_t = now
        button_state.gallery_viewer_swipe_last_x = x
        button_state.gallery_viewer_swipe_drag_moved = False
        button_state.gallery_viewer_swipe_inertia_active = False
        return True

    if event == cv2.EVENT_MOUSEMOVE and button_state.gallery_viewer_swipe_dragging:
        dx = x - button_state.gallery_viewer_swipe_start_x
        if abs(dx) > ui_cache.DRAG_PX:
            button_state.gallery_viewer_swipe_drag_moved = True
        # Clamp and apply rubber band at first/last item
        max_offset = 120.0
        raw = max(-max_offset, min(max_offset, float(dx)))
        button_state.gallery_viewer_swipe_offset = _viewer_rubber_band_offset(raw, idx, n)
        dt = max(1e-6, now - button_state.gallery_viewer_swipe_last_t)
        inst_v = (x - button_state.gallery_viewer_swipe_last_x) / dt * ui_cache.VIEWER_SWIPE_FLING_GAIN
        button_state.gallery_viewer_swipe_velocity = 0.5 * button_state.gallery_viewer_swipe_velocity + 0.5 * inst_v
        button_state.gallery_viewer_swipe_last_t = now
        button_state.gallery_viewer_swipe_last_x = x
        return True

    return False


def handle_gallery_mouse(event, x, y, flags, output_dir) -> bool:
    if not button_state.gallery_open or button_state.gallery_viewer_mode != "grid":
        return False

    now = time.perf_counter()

    if event == cv2.EVENT_LBUTTONDOWN:
        button_state.gallery_dragging = True
        button_state.gallery_drag_start_y = y
        button_state.gallery_drag_start_x = x
        button_state.gallery_drag_start_scroll = button_state.gallery_scroll_offset
        button_state.gallery_drag_moved = False

        button_state.gallery_scroll_velocity = 0.0
        button_state.gallery_last_drag_t = now
        button_state.gallery_last_drag_y = y
        button_state.gallery_inertia_active = False
        return True

    if event == cv2.EVENT_MOUSEMOVE and button_state.gallery_dragging:
        dy = y - button_state.gallery_drag_start_y
        dx = x - button_state.gallery_drag_start_x

        if (abs(dy) > ui_cache.DRAG_PX) or (abs(dx) > ui_cache.DRAG_PX):
            button_state.gallery_drag_moved = True

        if button_state.gallery_drag_moved:
            new_scroll = button_state.gallery_drag_start_scroll - dy
            max_scroll = int(getattr(button_state, "gallery_max_scroll", 0))
            button_state.gallery_scroll_offset = max(0, min(int(new_scroll), max_scroll))

            dt = max(1e-6, now - button_state.gallery_last_drag_t)
            FLING_GAIN = 2
            EMA_ALPHA = 0.6

            inst_v = (button_state.gallery_last_drag_y - y) / dt
            inst_v *= FLING_GAIN

            button_state.gallery_scroll_velocity = (1.0 - EMA_ALPHA) * button_state.gallery_scroll_velocity + EMA_ALPHA * inst_v
            button_state.gallery_last_drag_t = now
            button_state.gallery_last_drag_y = y

        return True

    if event == cv2.EVENT_LBUTTONUP and button_state.gallery_dragging:
        button_state.gallery_dragging = False

        if not button_state.gallery_drag_moved:
            return handle_gallery_click(x, y, output_dir)

        if abs(button_state.gallery_scroll_velocity) > 50.0:
            button_state.gallery_inertia_active = True
            button_state.gallery_last_inertia_t = now
        else:
            button_state.gallery_inertia_active = False
            button_state.gallery_scroll_velocity = 0.0

        return True

    return False
