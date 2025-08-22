# -*- coding: utf-8 -*-
import os
import sys
import re
import srt
from datetime import timedelta
from pathlib import Path

# ปรับ environment ก่อน import โมดูลที่อาจใช้ GPU/CUDA (ปิด GPU ถ้าต้องการบังคับให้ใช้ CPU)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    # Qt Widgets
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QFileDialog, QTextEdit, QCheckBox,
        QListWidget, QInputDialog, QMessageBox, QLineEdit, QTabWidget,
        QSpinBox, QSlider
    )
    from PySide6.QtCore import Qt

    # SRT handling

    # PyThaiNLP
    from pythainlp.util import thaiword_to_num
    from pythainlp.tokenize import word_tokenize, sent_tokenize
    from pythainlp.spell import correct as spell_correct
    from pythainlp.corpus.common import thai_words  # ชุดคำผสม
    # NER อาจเป็น heavy model -> ใช้แบบ try/except และให้ user ปิดได้
    from pythainlp.tag.named_entity import NER

except ImportError as e:
    print(f"เกิดข้อผิดพลาด: ไม่พบไลบรารีที่จำเป็น -> {e}")
    print("กรุณาติดตั้งด้วยคำสั่ง: pip install PySide6 pythainlp[attacut] srt")
    sys.exit(1)

# สร้าง NER instance อย่างระมัดระวัง (บาง environment จะ error ถ้าโมเดลโหลดผิด)
try:
    ner = NER()
except Exception as e:
    ner = None
    print("WARN: NER โหลดไม่ได้, จะทำงานในโหมดไม่มี NER. สาเหตุ:", e)

# ====== พจนานุกรม / ชุดคำพื้นฐาน (ค่าเริ่มต้น) ======
conj_words_before_default = [
    "ถ้าไม่อย่างงั้น", "ซึ่งจริงจริงแล้ว", "แต่ถ้า", "ดังนั้น", "เพราะว่า", "แม้ว่า", "อย่างไรก็ตาม",
    "ลอง", "ก็", "ใน", "ขึ้นมา", "คน", "ที่", "จาก", "แต่", "และ", "หรือ", "ถ้า", "เมื่อ", "ซึ่ง",
    "เพราะ", "ดังนั้น", "ก็จะ", "ก็คือ", "ตัวอย่าง", "แล้วก็", "ทำให้", "จะ", "เปอร์เซ็นต์", "แสดง"
]

conj_words_after_default = [
    "ส่วน", "อายุ", "จะเป็น", "เพิ่มเข้ามา", "ราคา", "ยังไง", "เริ่มที่", "คอลัม", "คิดเป็น", "อย่างเช่น",
    "จริง", "ทั้งหมด", "ร้อยละ", "จะเป็น", "ที่", "ปี", "จุด", "เปอร์เซ็นต์", "เข้ามา"
]

remove_words_base_default = [
    "ครับ", "ค่ะ", "นะครับ", "นะคะ", "ค่ะ", "ครับ", "อะ", "เนี้ย", "เนี่ย", "เอี้ย", "เอี่ย", "มัน", "นะ"
]

base_number_units_default = [
    "บาท", "ปี", "ชิ้น", "คน", "ลูก", "ตัว", "แผ่น", "กล่อง", "ลิตร", "เมตร",
    "กิโลกรัม", "กิโล", "เปอร์เซ็นต์", "เช่น", "กับ", "ถึง",
    "ซอย", "ถนน", "ตำบล", "อำเภอ", "จังหวัด", "มณฑล", "ประเทศ",
    "เขตพื้นที่", "ภูมิภาค", "เมือง", "เดือน", "วัน"
]

# โหลดชุดคำผสม (คำไทยที่มีใน corpus)
try:
    compound_words_set = set(w for w in thai_words() if len(w) > 1)
except Exception:
    compound_words_set = set()

# ----------------- ฟังก์ชันช่วยเหลือ (จากโปรแกรมที่ 1) -----------------
def is_part_of_compound(word, compound_set):
    for cw in compound_set:
        if word in cw and len(cw) > len(word):
            return True
    return False

def extract_named_entities_safe(text, use_ner=True):
    if not use_ner or ner is None:
        return set()
    try:
        entities = ner.tag(text)
        ne_set = set()
        cur, cur_tag = [], None
        for w, tag in entities:
            if tag.startswith("B-"):
                if cur:
                    ne_set.add(''.join(cur))
                cur = [w]
                cur_tag = tag[2:]
            elif tag.startswith("I-") and cur_tag == tag[2:]:
                cur.append(w)
            else:
                if cur:
                    ne_set.add(''.join(cur))
                cur, cur_tag = [], None
        if cur:
            ne_set.add(''.join(cur))
        return ne_set
    except Exception as e:
        print("WARN: NER.tag failed:", e)
        return set()

def insert_spaces_around_units(text, units, compound_set, user_entities):
    all_skip_words = set(compound_set).union(set(user_entities))
    for unit in units:
        if any(unit in w and len(w) > len(unit) for w in all_skip_words):
            continue
        text = re.sub(rf'(?<!\s)(?<!^){re.escape(unit)}', f' {unit}', text)
        text = re.sub(rf'{re.escape(unit)}(?!\s|$)', f'{unit} ', text)
    return text

def custom_sentence_split(text, conj_before, conj_after, unit_list, compound_set, user_entities):
    text = insert_spaces_around_units(text, unit_list, compound_set, user_entities)
    tokens = word_tokenize(text, engine="attacut")
    new_tokens = []
    for i, token in enumerate(tokens):
        if token in conj_before and i > 0 and not new_tokens[-1].endswith(' '):
            new_tokens.append(' ')
        new_tokens.append(token)
        if token in conj_after and i + 1 < len(tokens) and not tokens[i + 1].startswith(' '):
            new_tokens.append(' ')
    text = ''.join(new_tokens)
    try:
        sentences = sent_tokenize(text, engine='crfcut')
    except Exception:
        sentences = re.split(r'(?<=[\.\?\!]|[。])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def custom_detokenize(tokens: list) -> str:
    if not tokens: return ""
    built = tokens[0]
    for i in range(1, len(tokens)):
        is_current_ascii = all(ord(c) < 128 for c in tokens[i])
        is_prev_ascii = all(ord(c) < 128 for c in tokens[i-1])
        if is_current_ascii != is_prev_ascii:
            built += ' '
        built += tokens[i]
    return built

def find_thai_number_phrases(text):
    pattern = r'(ศูนย์|หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า|สิบ|ร้อย|พัน|หมื่น|แสน|ล้าน|และ|ยี่|เอ็ด|ยี่สิบ|แสน|ล้าน)+'
    return [(m.start(), m.end(), m.group()) for m in re.finditer(pattern, text)]

def convert_thai_numbers_in_text(text):
    matches = find_thai_number_phrases(text)
    if not matches: return text
    new_text, last_idx = "", 0
    for start, end, phrase in matches:
        new_text += text[last_idx:start]
        try:
            new_text += str(thaiword_to_num(phrase))
        except Exception:
            new_text += phrase
        last_idx = end
    new_text += text[last_idx:]
    return new_text

def safe_spell_correct(text, max_len=500):
    if not text or len(text) <= max_len:
        try:
            return spell_correct(text)
        except:
            return text
    tokens = word_tokenize(text, engine="attacut")
    chunks, cur, cur_len = [], [], 0
    for t in tokens:
        cur.append(t)
        cur_len += len(t)
        if cur_len > max_len:
            part = ''.join(cur)
            try:
                corrected = spell_correct(part)
            except:
                corrected = part
            chunks.append(corrected)
            cur, cur_len = [], 0
    if cur:
        part = ''.join(cur)
        try:
            corrected = spell_correct(part)
        except:
            corrected = part
        chunks.append(corrected)
    return ''.join(chunks)

def replace_user_corrections(text, user_corrections):
    if not user_corrections: return text
    for old_w, new_w in user_corrections.items():
        pattern = rf'(?<![\w\d]){re.escape(old_w)}(?![\w\d])'
        try:
            text = re.sub(pattern, new_w, text)
        except Exception:
            pass
    return text

def process_srt_content_v2(srt_content, loanword_dict, words_to_remove, should_convert_num, should_correct_spell,
                           conj_before, conj_after, unit_list, compound_set, user_entities, user_corrections, use_ner=True):
    try:
        srt_blocks = list(srt.parse(srt_content))
    except Exception:
        return srt_content

    for block in srt_blocks:
        processed_text = " ".join(block.content.split())
        if user_corrections:
            processed_text = replace_user_corrections(processed_text, user_corrections)

        extracted_entities = extract_named_entities_safe(processed_text, use_ner=use_ner)
        all_entities = set(user_entities).union(extracted_entities)
        sentences = custom_sentence_split(processed_text, conj_before, conj_after, unit_list, compound_set, all_entities)

        processed_sentences = []
        for sentence in sentences:
            if should_convert_num:
                sentence = convert_thai_numbers_in_text(sentence)
            if should_correct_spell:
                sentence = safe_spell_correct(sentence)
            processed_sentences.append(sentence)

        final_text = ' '.join(" ".join(processed_sentences).split())
        block.content = final_text

    return srt.compose(srt_blocks)

def detect_suspicious_words(text, loanword_dict, compound_set, user_entities):
    try:
        tokens = word_tokenize(text, engine="attacut")
    except Exception:
        tokens = text.split()
    suspicious = []
    for t in tokens:
        low = t.lower()
        if t and t.strip() and low != 'srt' and low not in loanword_dict and \
           t not in compound_set and t not in user_entities and not re.match(r'^[0-9]+$', t) and len(t) > 1:
            suspicious.append(t)
    return sorted(set(suspicious))

def find_error_sentences(text, loanword_dict, compound_set, user_entities):
    """
    Return a list of sentences that contain suspicious words.
    """
    error_sentences = []
    sentences = sent_tokenize(text, engine="crfcut")
    for sent in sentences:
        suspicious = detect_suspicious_words(sent, loanword_dict, compound_set, user_entities)
        if suspicious:
            error_sentences.append((sent, suspicious))
    return error_sentences


# ----------------- ฟังก์ชันช่วยเหลือ (จากโปรแกรมที่ 2) -----------------
def split_text_by_chars(text, max_chars):
    words = text.split()
    lines, current_line = [], ""
    for w in words:
        if len(current_line) + len(w) + (1 if current_line else 0) <= max_chars:
            current_line += (" " if current_line else "") + w
        else:
            if current_line: lines.append(current_line)
            current_line = w
    if current_line: lines.append(current_line)
    return "\n".join(lines)

def parse_srt_splitter(content):
    pattern = re.compile(r"(\d+)\r?\n([\d:,]+)\s*-->\s*([\d:,]+)\r?\n(.*?)(?=\r?\n\r?\n\d+|\Z)", re.S)
    entries = []
    for m in pattern.finditer(content):
        idx, start, end, text = m.groups()
        text = re.sub(r'[\r\n]+', ' ', text).strip()
        entries.append({"index": int(idx), "start": timecode_to_ms(start), "end": timecode_to_ms(end), "text": text})
    return entries

def timecode_to_ms(tc):
    parts = tc.split(':')
    s, ms = parts[2].split(',')
    return (int(parts[0]) * 3600 + int(parts[1]) * 60 + int(s)) * 1000 + int(ms)

def ms_to_timecode(ms):
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def rebuild_srt(entries, max_chars, divide_time):
    new_entries, counter = [], 1
    for e in entries:
        lines = split_text_by_chars(e['text'], max_chars).split('\n')
        if not lines or (len(lines) == 1 and not lines[0]): continue
        
        if not divide_time:
            for line in lines:
                new_entries.append({"index": counter, "start": e['start'], "end": e['end'], "text": line})
                counter += 1
        else:
            duration = e['end'] - e['start']
            part_dur = duration // len(lines) if len(lines) > 0 else 0
            for i, line in enumerate(lines):
                start_time = e['start'] + i * part_dur
                end_time = e['start'] + (i + 1) * part_dur if i < len(lines) - 1 else e['end']
                new_entries.append({"index": counter, "start": start_time, "end": end_time, "text": line})
                counter += 1
    return new_entries

def srt_entries_to_text(entries):
    out = []
    for e in entries:
        out.append(f'{e["index"]}\n{ms_to_timecode(e["start"])} --> {ms_to_timecode(e["end"])}\n{e["text"]}')
    return "\n\n".join(out)


# =================================================================
#  คลาส GUI สำหรับ แท็บที่ 1: Batch SRT Processor
# =================================================================
class BatchSrtProcessorTab(QWidget):
    def __init__(self):
        super().__init__()
        self.user_corrections = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("ยังไม่ได้เลือกโฟลเดอร์")
        self.folder_label.setStyleSheet("border:1px solid gray; padding:6px;")
        btn = QPushButton("เลือกโฟลเดอร์ SRT")
        btn.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.folder_label, stretch=1)
        folder_layout.addWidget(btn)
        layout.addLayout(folder_layout)

        settings_layout = QHBoxLayout()
        
        # --- UI ด้านซ้าย ---
        left_tabs = QTabWidget()
        tab_replace = QWidget()
        tab_replace_layout = QVBoxLayout(tab_replace)
        tab_replace_layout.addWidget(QLabel("คำทับศัพท์ (รูปแบบ: คำไทย:คำอังกฤษ)"))
        self.loanword_edit = QTextEdit()
        tab_replace_layout.addWidget(self.loanword_edit)
        tab_replace_layout.addWidget(QLabel("คำเฉพาะ (เพิ่มเอง 1 ต่อบรรทัด)"))
        self.entities_edit = QTextEdit()
        tab_replace_layout.addWidget(self.entities_edit)
        left_tabs.addTab(tab_replace, "การแทนที่คำ")
        
        tab_split = QWidget()
        tab_split_layout = QVBoxLayout(tab_split)
        tab_split_layout.addWidget(QLabel("คำที่ต้องการลบ (1 ต่อบรรทัด)"))
        self.remove_edit = QTextEdit('\n'.join(remove_words_base_default))
        tab_split_layout.addWidget(self.remove_edit)
        tab_split_layout.addWidget(QLabel("คำตัดประโยค (เว้นวรรคก่อนหน้า)"))
        self.conj_before_edit = QTextEdit('\n'.join(conj_words_before_default))
        tab_split_layout.addWidget(self.conj_before_edit)
        tab_split_layout.addWidget(QLabel("คำตัดประโยค (เว้นวรรคด้านหลัง)"))
        self.conj_after_edit = QTextEdit('\n'.join(conj_words_after_default))
        tab_split_layout.addWidget(self.conj_after_edit)
        left_tabs.addTab(tab_split, "การลบและตัดคำ")
        
        tab_units = QWidget()
        tab_units_layout = QVBoxLayout(tab_units)
        tab_units_layout.addWidget(QLabel("หน่วยนับ (เว้นวรรครอบๆ)"))
        self.units_edit = QTextEdit('\n'.join(base_number_units_default))
        tab_units_layout.addWidget(self.units_edit)
        left_tabs.addTab(tab_units, "หน่วยนับ")
        settings_layout.addWidget(left_tabs, stretch=3)

        # --- UI ด้านขวา ---
        right = QVBoxLayout()
        self.ner_check = QCheckBox("ใช้ NER (อาจช้า / อาจโหลดโมเดลล้มเหลว)")
        self.ner_check.setChecked(ner is not None)
        self.ner_check.setEnabled(ner is not None)
        right.addWidget(self.ner_check)
        self.spell_check = QCheckBox("แก้คำผิดอัตโนมัติ (safe)")
        right.addWidget(self.spell_check)
        self.num_check = QCheckBox("แปลงคำอ่านตัวเลขเป็นเลขอารบิก")
        right.addWidget(self.num_check)
        right.addWidget(QLabel("คำที่น่าสงสัย (ดับเบิลคลิกเพื่อแก้)"))
        self.suspicious_list = QListWidget()
        self.suspicious_list.itemDoubleClicked.connect(self.edit_suspicious)
        right.addWidget(self.suspicious_list, stretch=1)
        right.addWidget(QLabel("Quick add user correction (old:new)"))
        self.quick_line = QLineEdit()
        self.quick_line.setPlaceholderText("อันซักเจอร์เดต้า:Unstructured Data")
        quick_btn = QPushButton("เพิ่มใน corrections")
        quick_btn.clicked.connect(self.quick_add_correction)
        right.addWidget(self.quick_line)
        right.addWidget(quick_btn)
        settings_layout.addLayout(right, stretch=2)

        layout.addLayout(settings_layout, stretch=2)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(QLabel("Log สถานะ"))
        layout.addWidget(self.log_box, stretch=1)

        bottom = QHBoxLayout()
        self.detect_btn = QPushButton("1. วิเคราะห์หาคำที่น่าสงสัย")
        self.detect_btn.clicked.connect(self.run_detect_all)
        bottom.addWidget(self.detect_btn)
        self.process_btn = QPushButton("2. เริ่มประมวลผลไฟล์ทั้งหมด")
        self.process_btn.clicked.connect(self.process_all_files)
        bottom.addWidget(self.process_btn)
        self.load_btn = QPushButton("โหลด mapping")
        self.load_btn.clicked.connect(self.load_corrections_from_file)
        bottom.addWidget(self.load_btn)
        self.save_btn = QPushButton("บันทึก mapping")
        self.save_btn.clicked.connect(self.save_corrections_to_file)
        bottom.addWidget(self.save_btn)
        layout.addLayout(bottom)

    def log(self, *args):
        s = " ".join(str(a) for a in args)
        self.log_box.append(s)
        print(s) # ยังคง print ไว้สำหรับ debug ผ่าน console

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์ไฟล์ SRT")
        if folder:
            self.folder_label.setText(folder)
            self.log(f"เลือกโฟลเดอร์: {folder}")

    def quick_add_correction(self):
        txt = self.quick_line.text().strip()
        if ':' in txt:
            old, new = [x.strip() for x in txt.split(':', 1)]
            if old:
                self.user_corrections[old] = new
                self.log(f"เพิ่ม correction: {old} → {new}")
                self.update_suspicious_list(old, new)
                self.quick_line.clear()
        else:
            QMessageBox.warning(self, "รูปแบบผิด", "รูปแบบต้องเป็น old:new")

    def edit_suspicious(self, item):
        text = item.text()
        old = text.split('→')[0].strip() if '→' in text else text.strip()
        current_val = self.user_corrections.get(old, old)
        new, ok = QInputDialog.getText(self, "แก้คำ", f"แก้คำ '{old}' เป็น:", text=current_val)
        if ok and new.strip():
            self.user_corrections[old] = new.strip()
            item.setText(f"{old} → {new.strip()}")
            self.log(f"แก้คำ mapping: {old} → {new.strip()}")

    def update_suspicious_list(self, old_word, new_word):
        """Update or add an entry in the suspicious words list with the new correction."""
        for i in range(self.suspicious_list.count()):
            item = self.suspicious_list.item(i)
            if item.text().startswith(old_word + " →") or item.text() == old_word:
                item.setText(f"{old_word} → {new_word}")
                return
        self.suspicious_list.addItem(f"{old_word} → {new_word}")
    def run_detect_all(self):
        """
        Analyze all SRT files in the selected folder to detect suspicious words for user correction.
        """
    def run_detect_all(self):
        folder = self.folder_label.text()
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "ต้องเลือกโฟลเดอร์", "กรุณาเลือกโฟลเดอร์ที่ถูกต้องก่อน")
            return
        
        srt_files = list(Path(folder).glob("*.srt"))
        if not srt_files:
            QMessageBox.warning(self, "ไม่พบไฟล์", "ไม่พบไฟล์ .srt ในโฟลเดอร์")
            return

        loanword_dict = {a.strip().lower(): b.strip() for line in self.loanword_edit.toPlainText().splitlines() if ':' in line for a, b in [line.split(':', 1)]}
        user_entities = {line.strip() for line in self.entities_edit.toPlainText().splitlines() if line.strip()}
        
        self.suspicious_list.clear()
        self.user_corrections.clear()
        self.log("--- เริ่มการวิเคราะห์หาคำที่น่าสงสัย ---")

        combined_susp = set()
        for srt_file in srt_files:
            try:
                raw_content = srt_file.read_text(encoding='utf-8')
                processed = process_srt_content_v2(
                    raw_content, loanword_dict, all_remove, self.num_check.isChecked(),
                    self.spell_check.isChecked(), all_conj_before, all_conj_after,
                    all_units, compound_words_set, user_entities, self.user_corrections,
                    use_ner=self.ner_check.isChecked()
                )
                out_path = output_folder / srt_file.name
                out_path.write_text(processed, encoding='utf-8')
                self.log(f"✅ ประมวลผลไฟล์ {srt_file.name} สำเร็จ")
            except Exception as e:
                self.log(f"❌ ประมวลผลไฟล์ {srt_file.name} ล้มเหลว: {e}")
        
        for w in sorted(combined_susp): self.suspicious_list.addItem(w)
        self.log(f"ตรวจเสร็จ พบคำที่น่าสงสัย {len(combined_susp)} คำ")
        if combined_susp:
            QMessageBox.information(self, "ตรวจพบคำ", "พบคำที่น่าสงสัย กรุณาตรวจสอบและแก้ไขในรายการด้านขวา")

    def load_corrections_from_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "โหลดไฟล์ mapping", "", "Text Files (*.txt)")
        if not path: return
        count = 0
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() and not line.strip().startswith('#') and ':' in line:
                        old, new = [x.strip() for x in line.split(':', 1)]
                        if old:
                            self.user_corrections[old] = new
                            self.update_suspicious_list(old, new)
                            count += 1
            self.log(f"✅ โหลด mapping จากไฟล์ '{Path(path).name}' สำเร็จ, {count} รายการ")
            QMessageBox.information(self, "โหลดสำเร็จ", f"โหลด mapping สำเร็จ {count} รายการ")
        except Exception as e:
            self.log(f"❌ เกิดข้อผิดพลาดในการโหลดไฟล์ mapping: {e}")
            QMessageBox.critical(self, "เกิดข้อผิดพลาด", f"ไม่สามารถโหลดไฟล์ mapping ได้\n{e}")

    def save_corrections_to_file(self):
        if not self.user_corrections:
            QMessageBox.warning(self, "ไม่มี mapping", "ไม่มี mapping ให้บันทึก")
            return
        path, _ = QFileDialog.getSaveFileName(self, "บันทึก mapping", "", "Text Files (*.txt)")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                for k, v in sorted(self.user_corrections.items()):
                    f.write(f"{k}:{v}\n")
            self.log(f"บันทึก mapping {len(self.user_corrections)} รายการ เรียบร้อย: {path}")

    def process_all_files(self):
        """
        Process all SRT files in the selected folder using the current settings and save the results to an output folder.
        """
        folder = self.folder_label.text()
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "ต้องเลือกโฟลเดอร์", "กรุณาเลือกโฟลเดอร์ที่ถูกต้องก่อน")
            return

        srt_files = list(Path(folder).glob("*.srt"))
        if not srt_files:
            QMessageBox.warning(self, "ไม่พบไฟล์", "ไม่พบไฟล์ .srt ในโฟลเดอร์")
            return

        # อ่านค่าทั้งหมดจาก UI ณ เวลาที่กดปุ่ม
        loanword_dict = {a.strip().lower(): b.strip() for line in self.loanword_edit.toPlainText().splitlines() if ':' in line for a, b in [line.split(':', 1)]}
        user_entities = {line.strip() for line in self.entities_edit.toPlainText().splitlines() if line.strip()}
        all_remove = {line.strip() for line in self.remove_edit.toPlainText().splitlines() if line.strip()}
        self.log(f"DEBUG: คำที่จะถูกลบ (จาก UI) คือ: {all_remove}")
        all_conj_before = [line.strip() for line in self.conj_before_edit.toPlainText().splitlines() if line.strip()]
        all_conj_after = [line.strip() for line in self.conj_after_edit.toPlainText().splitlines() if line.strip()]
        all_units = [line.strip() for line in self.units_edit.toPlainText().splitlines() if line.strip()]

        output_folder = Path(folder) / "output_processed"
        output_folder.mkdir(exist_ok=True)
        self.log(f"--- เริ่มประมวลผล {len(srt_files)} ไฟล์ -> โฟลเดอร์: {output_folder} ---")

        for srt_file in srt_files:
            try:
                content = srt_file.read_text(encoding='utf-8')

                # --- VVV จุดที่แก้ไข VVV ---
                processed = process_srt_content_v2(
                    srt_content=content,
                    loanword_dict=loanword_dict,
                    words_to_remove=all_remove,
                    should_convert_num=self.num_check.isChecked(),
                    should_correct_spell=self.spell_check.isChecked(),
                    conj_before=all_conj_before,
                    conj_after=all_conj_after,
                    unit_list=all_units,
                    compound_set=compound_words_set,
                    user_entities=user_entities,
                    user_corrections=self.user_corrections,
                    use_ner=self.ner_check.isChecked()
                )
                # --- ^^^ จุดที่แก้ไข ^^^ ---

                out_path = output_folder / srt_file.name
                out_path.write_text(processed, encoding='utf-8')
                self.log(f"✅ ประมวลผลไฟล์: {srt_file.name} -> {out_path}")
            except Exception as e:
                self.log(f"❌ ประมวลผลไฟล์ {srt_file.name} ผิดพลาด: {e}")

        QMessageBox.information(self, "เสร็จสิ้น", f"ประมวลผลไฟล์ทั้งหมดเสร็จสิ้น\nผลลัพธ์อยู่ในโฟลเดอร์ 'output_processed'")
        self.log("🎉 ประมวลผลไฟล์ทั้งหมดเสร็จสิ้น 🎉")
        
# =================================================================
#  คลาส GUI สำหรับ แท็บที่ 2: SRT Splitter
# =================================================================
class SRTSplitterTab(QWidget):
    def __init__(self, processor_tab_ref):
        super().__init__()
        self.processor_tab = processor_tab_ref  # รับ reference ของแท็บแรกเข้ามา
        self.setWindowTitle("SRT Jump Cut Helper")
        self.srt_path = None
        self.entries = []
        self.processed_entries = []
        self.preview_index = -1
        self.slider_is_moving = False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # --- UI Elements ---
        top_buttons_layout = QHBoxLayout()
        btn_load_from_proc = QPushButton("โหลดผลลัพธ์จากขั้นตอนที่ 1")
        btn_load_from_proc.setStyleSheet("background-color: #cce5ff; padding: 5px;")
        btn_load_from_proc.clicked.connect(self.load_from_processor)
        top_buttons_layout.addWidget(btn_load_from_proc)
        
        btn_file = QPushButton("หรือเลือกไฟล์ SRT อื่นๆ")
        btn_file.clicked.connect(self.choose_file_dialog)
        top_buttons_layout.addWidget(btn_file)
        layout.addLayout(top_buttons_layout)

        self.file_label = QLabel("ยังไม่ได้เลือกไฟล์ SRT")
        self.file_label.setStyleSheet("border: 1px solid gray; padding: 5px;")
        layout.addWidget(self.file_label)

        char_layout = QHBoxLayout()
        char_layout.addWidget(QLabel("จำนวนตัวอักษรสูงสุดต่อ 1 Caption ย่อย:"))
        self.char_spin = QSpinBox()
        self.char_spin.setRange(10, 100)
        self.char_spin.setValue(40)
        char_layout.addWidget(self.char_spin)
        layout.addLayout(char_layout)

        self.divide_time_cb = QCheckBox("แบ่งเวลาย่อยอัตโนมัติ (สำหรับทำ Jump Cut)")
        self.divide_time_cb.setChecked(True)
        layout.addWidget(self.divide_time_cb)

        btn_run = QPushButton("ประมวลผลและดู Preview")
        btn_run.clicked.connect(self.run_preview)
        layout.addWidget(btn_run)

        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("ก่อนหน้า")
        self.btn_prev.clicked.connect(self.show_prev_sub)
        nav_layout.addWidget(self.btn_prev)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.slider_moved)
        nav_layout.addWidget(self.slider)

        self.btn_next = QPushButton("ถัดไป")
        self.btn_next.clicked.connect(self.show_next_sub)
        nav_layout.addWidget(self.btn_next)
        layout.addLayout(nav_layout)
        self.update_nav_buttons()

        btn_save = QPushButton("บันทึกไฟล์ .srt ใหม่")
        btn_save.clicked.connect(self.save_file)
        layout.addWidget(btn_save)

        self.preview_label = QLabel("Preview ซับจะขึ้นที่นี่")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet("font-size: 18px; border: 1px solid gray; min-height: 80px; padding: 10px;")
        layout.addWidget(self.preview_label, stretch=1)

    def load_from_processor(self):
        source_folder = self.processor_tab.folder_label.text()
        if not os.path.isdir(source_folder):
            QMessageBox.warning(self, "โฟลเดอร์ไม่ถูกต้อง", "กรุณาไปที่แท็บ 1 และเลือกโฟลเดอร์ต้นทางก่อน")
            return
            
        output_folder = Path(source_folder) / "output_processed"
        if not output_folder.exists():
            QMessageBox.warning(self, "ไม่พบผลลัพธ์", "ไม่พบโฟลเดอร์ 'output_processed'\nกรุณากด 'เริ่มประมวลผล' ในแท็บที่ 1 ก่อน")
            return
            
        path, _ = QFileDialog.getOpenFileName(self, "เลือกไฟล์ที่ประมวลผลแล้ว", str(output_folder), "SRT Files (*.srt)")
        if path:
            self.process_selected_file(path)

    def choose_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "เลือกไฟล์ SRT", "", "SRT Files (*.srt)")
        if path:
            self.process_selected_file(path)

    def process_selected_file(self, path):
        try:
            self.srt_path = path
            self.file_label.setText(Path(path).name)
            content = Path(path).read_text(encoding="utf-8")
            self.entries = parse_srt_splitter(content)
            if not self.entries:
                QMessageBox.warning(self, "Error", "ไม่พบซับในไฟล์นี้")
                self.reset_preview()
                return
            self.preview_label.setText(f"โหลดไฟล์เรียบร้อย ({len(self.entries)} captions) รอประมวลผล")
            self.reset_preview()
        except Exception as e:
            QMessageBox.critical(self, "File Read Error", f"ไม่สามารถอ่านไฟล์ได้:\n{e}")
            self.reset_ui()

    def reset_ui(self):
        self.srt_path = None
        self.file_label.setText("ยังไม่ได้เลือกไฟล์")
        self.reset_preview()

    def reset_preview(self):
        self.processed_entries = []
        self.preview_index = -1
        self.slider.setEnabled(False)
        self.update_nav_buttons()
        QApplication.processEvents()

    def run_preview(self):
        if not self.srt_path:
            QMessageBox.warning(self, "Error", "กรุณาเลือกไฟล์ SRT ก่อน")
            return
        self.processed_entries = rebuild_srt(self.entries, self.char_spin.value(), self.divide_time_cb.isChecked())
        if not self.processed_entries:
            QMessageBox.warning(self, "Error", "ไม่พบซับในไฟล์ หรือประมวลผลล้มเหลว")
            return
        
        self.preview_label.setText(f"ประมวลผลเสร็จสิ้น (ได้ {len(self.processed_entries)} captions ย่อย)")
        QApplication.processEvents()
        
        self.preview_index = 0
        self.slider.setEnabled(True)
        self.slider.setRange(0, len(self.processed_entries) - 1)
        self.show_sub(self.preview_index)
        self.update_nav_buttons()

    def show_sub(self, index):
        if 0 <= index < len(self.processed_entries):
            sub = self.processed_entries[index]
            self.preview_label.setText(sub["text"])
            self.preview_index = index
            if not self.slider_is_moving:
                self.slider.setValue(index)
        else:
            self.preview_label.setText("")

    def show_next_sub(self):
        if self.preview_index < len(self.processed_entries) - 1:
            self.show_sub(self.preview_index + 1)
            self.update_nav_buttons()

    def show_prev_sub(self):
        if self.preview_index > 0:
            self.show_sub(self.preview_index - 1)
            self.update_nav_buttons()

    def slider_moved(self, value):
        self.slider_is_moving = True
        self.show_sub(value)
        self.update_nav_buttons()
        self.slider_is_moving = False

    def update_nav_buttons(self):
        has_entries = bool(self.processed_entries)
        self.slider.setEnabled(has_entries)
        self.btn_prev.setEnabled(has_entries and self.preview_index > 0)
        self.btn_next.setEnabled(has_entries and self.preview_index < len(self.processed_entries) - 1)

    def save_file(self):
        if not self.processed_entries:
            QMessageBox.warning(self, "Error", "กรุณาประมวลผลก่อนบันทึก")
            return
        
        original_path = Path(self.srt_path)
        default_name = f"{original_path.stem}_split.srt"
        default_path = original_path.with_name(default_name)
        out_path, _ = QFileDialog.getSaveFileName(self, "บันทึกไฟล์ SRT ใหม่", str(default_path), "SRT Files (*.srt)")

        if out_path:
            try:
                txt = srt_entries_to_text(self.processed_entries)
                Path(out_path).write_text(txt, encoding="utf-8")
                QMessageBox.information(self, "สำเร็จ", f"บันทึกไฟล์เรียบร้อยแล้วที่:\n{out_path}")
            except Exception as e:
                QMessageBox.critical(self, "File Write Error", f"ไม่สามารถบันทึกไฟล์ได้:\n{e}")

# =================================================================
#  คลาสหลักของโปรแกรม (Main Window)
# =================================================================
class SrtSuite(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SRT Suite - โปรแกรมจัดการซับไตเติ้ล")
        self.setGeometry(50, 50, 1100, 850)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        tab_widget = QTabWidget()

        # สร้างแท็บที่ 1 และเก็บ reference ไว้
        self.processor_tab = BatchSrtProcessorTab()
        
        # สร้างแท็บที่ 2 โดยส่ง reference ของแท็บแรกเข้าไป
        self.splitter_tab = SRTSplitterTab(processor_tab_ref=self.processor_tab)

        # เพิ่มแท็บทั้งสองเข้าไปใน Tab Widget
        tab_widget.addTab(self.processor_tab, "ขั้นตอนที่ 1: จัดการและแก้ซับ (Batch Process)")
        tab_widget.addTab(self.splitter_tab, "ขั้นตอนที่ 2: ตัดซับยาว (Jump Cut Splitter)")

        main_layout.addWidget(tab_widget)

# ----------------- main -----------------
def main():
    app = QApplication(sys.argv)
    window = SrtSuite()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()