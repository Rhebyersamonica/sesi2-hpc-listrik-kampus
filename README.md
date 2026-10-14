# Sesi 2: Menjembatani HPC untuk Masalah Lokal, Simulasi Kebijakan Listrik Kampus

**Nama:** (Rheby Ersa Monica) | **NPM:** (G1A023016) | Prodi Informatika, Universitas Bengkulu

Tugas Informatics Practitioner Lecture Series, *"Bridging Global Infrastructure to Address Local Challenge with HPC"*
(Dr. Worawan Diaz Carballo, Thammasat University, 5 Oktober 2026).

## Masalah lokal
Tagihan listrik kampus berasal dari perilaku pengguna gedung. Kebijakan AC (jam hidup, jam mati, suhu setelan) diuji dengan
simulasi berbasis agen (Monte Carlo) yang dijalankan **paralel**. Laboratorium wajib tetap menyala 24 jam (constraint dari pakar domain).

## Isi
| Bagian | Konsep dari materi |
|---|---|
| Mini-LINPACK | FLOPS dan benchmark TOP500 |
| Simulasi berbasis agen | Simulasi untuk masalah lokal (contoh Twin-B) |
| Komputasi paralel + strong scaling | Speedup dan efisiensi pada HPC |
| Analisis trade-off | Penghematan vs kenyamanan; keputusan akhir di manusia |
| `job.sbatch` | Contoh skrip batch SLURM untuk cluster HPC |

## Data
**Sintetis**, dibuat oleh kode sendiri (tidak ada unduhan). Tarif listrik dan pola hunian adalah asumsi yang bisa diubah di `hpc_case.py`.

## Cara menjalankan
```bash
python -m venv .venv
.venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python hpc_case.py --workers 4 --replicates 20
# di cluster SLURM: sbatch job.sbatch
```
Hasil tersimpan di folder `output/`. Catatan: speedup bergantung jumlah core komputermu.

## Hasil (simulasi 36 kebijakan × 20 replikasi, baseline ±19.900 kWh/bulan)
- Kebijakan paling hemat: AC hidup pukul 08.00–18.00 dengan setelan 26 °C → hemat ±51% (±Rp14,8 juta/bulan, tarif asumsi), ketidaknyamanan ±8%.
- Kebijakan lebih nyaman: AC hidup pukul 07.00–20.00 dengan setelan 26 °C → hemat ±46%, ketidaknyamanan ±1%.
- Ada trade-off hemat vs kenyamanan; pemilihan batas ketidaknyamanan adalah keputusan manusia (pengelola kampus).
- Uji *strong scaling*: jalankan tanpa `--skip-scaling`, lalu tulis speedup dan efisiensi dari komputermu (`output/02_scaling.png`).
- Tabel lengkap: `output/hasil_kebijakan.csv`; grafik: `output/03_tradeoff.png`.

## Keterbatasan
Data sintetis, model fisika AC disederhanakan, tarif listrik asumsi. Versi nyata butuh data hunian dan meteran listrik serta validasi pakar.
