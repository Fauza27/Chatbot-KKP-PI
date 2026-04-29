import pdfplumber
import sys
import os

PDF_PATH = r"C:\Users\Muhammad Fauza\penelitian-ilmiah\extract-pdf\Panduan Penyusunan Kuliah Kerja Praktik (KKP) Cetak.pdf"
OUTPUT_PATH = r"c:\Users\Muhammad Fauza\penelitian-ilmiah\hasil_ekstrak_panduan_kkp.txt"

def extract_text():
    all_text = []
    with pdfplumber.open(PDF_PATH) as pdf:
        total_pages = len(pdf.pages)
        print(f"Total halaman: {total_pages}")
        
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            text = page.extract_text()
            
            if text:
                all_text.append(f"{'='*60}")
                all_text.append(f"HALAMAN {page_num}")
                all_text.append(f"{'='*60}")
                all_text.append(text)
                all_text.append("")
            else:
                all_text.append(f"{'='*60}")
                all_text.append(f"HALAMAN {page_num} (tidak ada teks yang bisa diekstrak)")
                all_text.append(f"{'='*60}")
                all_text.append("")
            
            print(f"  Halaman {page_num}/{total_pages} selesai")
    
    full_text = "\n".join(all_text)
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    print(f"\nEkstraksi selesai! Output disimpan ke: {OUTPUT_PATH}")
    print(f"Total karakter: {len(full_text)}")

if __name__ == "__main__":
    extract_text()
