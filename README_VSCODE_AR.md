# تشغيل مشروع نقل الصورة باستخدام VS Code

## 1. تثبيت المتطلبات

افتح مجلد المشروع في VS Code، ثم افتح Terminal جديدًا من:

```text
Terminal → New Terminal
```

تأكد أن Python 3.10 أو أحدث مثبتة:

```bash
python --version
```

على Linux/macOS يمكن استخدام:

```bash
python3 --version
```

أنشئ virtual environment:

```bash
python -m venv .venv
```

فعّله:

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Windows CMD

```cmd
.venv\Scripts\activate.bat
```

### Linux/macOS

```bash
source .venv/bin/activate
```

ثبّت المكتبات:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. هيكل المشروع

```text
mars_coding_experiment/
├── data/
│   ├── source/
│   └── processed/
├── metadata/
├── results/
├── preprocess_image.py
├── serialize_image.py
├── hamming74.py
├── test_hamming74.py
├── ldpc.py
├── test_ldpc_structure.py
├── test_ldpc_decoder.py
├── simulate_bsc.py
├── run_final_experiment_restartable.py
└── analyze_final_results.py
```

## 3. تشغيل الاختبارات الأساسية

من داخل مجلد المشروع:

```bash
python preprocess_image.py
python serialize_image.py
python test_hamming74.py
python test_ldpc_structure.py
python test_ldpc_decoder.py
```

يجب أن ترى رسائل `PASS` في كل الاختبارات.

## 4. تشغيل اختبار الصورة بدون ضوضاء

```bash
python simulate_bsc.py
```

هذا الاختبار ينشئ:

```text
results/smoke_p0.csv
```

ويجب أن تكون النتيجة لكل الطرق:

```text
Decoded BER = 0
MSE = 0
SSIM = 1
Exact image = True
```

## 5. تشغيل التجربة النهائية

التجربة النهائية تستخدم:

```text
p = 0, 0.001, 0.005, 0.01, 0.02, 0.05
```

وتستخدم trial واحدًا عند `p=0` و100 trial لكل قيمة nonzero. لتشغيلها:

```bash
python run_final_experiment_restartable.py
```

الملف يحفظ النتائج تدريجيًا أثناء التشغيل في:

```text
results/final_bsc_metrics_restartable.csv
```

وهذا يعني أن النتائج المكتملة لا تضيع إذا أوقفت التشغيل. لا تغلق Terminal أثناء التشغيل إلا إذا كنت تريد إيقاف التجربة.

## 6. تحليل النتائج وإنشاء الأشكال

بعد اكتمال التجربة:

```bash
python analyze_final_results.py
```

ينتج:

```text
results/final_results_summary.csv
results/figures/decoded_ber_vs_p.png
results/figures/raw_decoded_ber_vs_p.png
results/figures/fer_vs_p.png
results/figures/psnr_vs_p.png
results/figures/ssim_vs_p.png
results/figures/decoder_time_vs_p.png
```

## 7. فتح النتائج في VS Code

من Explorer افتح:

```text
results/final_results_summary.csv
```

ولعرض الصور اضغط عليها داخل Explorer. ويمكن فتح ملف النتائج الخام:

```text
results/final_bsc_metrics_restartable.csv
```

## 8. تشغيل جزء واحد فقط للتجربة

لتشغيل تجربة واحدة من Python Interactive أو من Terminal:

```bash
python - <<'PY'
from simulate_bsc import load_bits, build_structure, run_trial
reference, bits = load_bits()
structure = build_structure()
row = run_trial("ldpc", reference, bits, p=0.01, seed=2026, structure=structure)
print(row)
PY
```

في Windows PowerShell يمكن بدلًا من ذلك إنشاء ملف باسم `one_trial.py` ووضع الكود داخله ثم تشغيل:

```bash
python one_trial.py
```

## 9. ملاحظات مهمة

- لا تغير ترتيب الـ bits أو crop أو resize إذا أردت إعادة إنتاج النتائج نفسها.
- لا تغير `MASTER_SEED` إذا أردت نفس noise realizations.
- نتيجة `p=0.05` تحتاج وقتًا أطول قليلًا بسبب زيادة عدد LDPC iterations.
- زمن decoding يعتمد على الجهاز وطريقة تشغيل Python، لذلك لا تقارنه كحقيقة عامة عن Hamming وLDPC.
- نتائج المشروع الحالية تخص صورة واحدة وقناة BSC ومصفوفة LDPC التعليمية المحددة في البحث.
